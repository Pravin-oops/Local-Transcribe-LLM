import os
import time
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QTabWidget, QTextBrowser,
                               QVBoxLayout, QWidget)

from localscribe import audio
from localscribe.config import load_settings, save_settings
from localscribe.jobs import Job, read_json
from localscribe.timefmt import hms
from localscribe.ui.settings_dialog import SettingsDialog
from localscribe.ui.worker import JobRunner

FILE_FILTER = "Audio and video ({});;All files (*)".format(
    " ".join(f"*{ext}" for ext in audio.AUDIO_EXTENSIONS))


class DropArea(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumHeight(90)
        self.setStyleSheet("DropArea { border: 2px dashed palette(mid); border-radius: 8px; }")
        layout = QVBoxLayout(self)
        self.title = QLabel("Drop an audio or video file here, or click Open file…")
        self.title.setAlignment(Qt.AlignCenter)
        font = self.title.font()
        font.setPointSize(font.pointSize() + 2)
        self.title.setFont(font)
        self.detail = QLabel("")
        self.detail.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title)
        layout.addWidget(self.detail)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LocalScribe")
        self.resize(1000, 720)
        self.setAcceptDrops(True)
        self.settings = load_settings()
        self.source = None
        self.job_dir = None
        self.started_at = None

        self.runner = JobRunner(self)
        self.runner.event.connect(self._on_event)
        self.runner.finished.connect(self._on_finished)

        self.drop = DropArea()

        self.open_btn = QPushButton("Open file…")
        self.start_btn = QPushButton("Start")
        self.cancel_btn = QPushButton("Cancel")
        self.settings_btn = QPushButton("Settings…")
        self.folder_btn = QPushButton("Open output folder")
        self.start_btn.setDefault(True)
        self.open_btn.clicked.connect(self._choose_file)
        self.start_btn.clicked.connect(self._start)
        self.cancel_btn.clicked.connect(self._cancel)
        self.settings_btn.clicked.connect(self._open_settings)
        self.folder_btn.clicked.connect(self._open_folder)

        buttons = QHBoxLayout()
        for b in (self.open_btn, self.start_btn, self.cancel_btn):
            buttons.addWidget(b)
        buttons.addStretch(1)
        self.model_label = QLabel()
        buttons.addWidget(self.model_label)
        buttons.addWidget(self.settings_btn)
        buttons.addWidget(self.folder_btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.status = QLabel("Ready")

        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        self.transcript = QPlainTextEdit(readOnly=True)
        self.transcript.setFont(mono)
        self.chapters = QTextBrowser(openExternalLinks=True)
        self.notes = QTextBrowser(openExternalLinks=True)
        self.log = QPlainTextEdit(readOnly=True)
        self.log.setFont(mono)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.transcript, "Transcript")
        self.tabs.addTab(self.chapters, "Chapters")
        self.tabs.addTab(self.notes, "Study Notes")
        self.tabs.addTab(self.log, "Log")

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self.drop)
        layout.addLayout(buttons)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(central)

        self._refresh_model_label()
        self._set_busy(False)

    # ------------------------------------------------------------------ file selection

    def dragEnterEvent(self, event):
        if not self.runner.is_running() and self._dropped_file(event):
            event.acceptProposedAction()

    def dropEvent(self, event):
        path = self._dropped_file(event)
        if path:
            self._set_source(path)

    @staticmethod
    def _dropped_file(event):
        urls = event.mimeData().urls()
        if len(urls) == 1 and urls[0].isLocalFile():
            path = Path(urls[0].toLocalFile())
            if path.is_file():
                return path
        return None

    def _choose_file(self):
        start = str(self.source.parent) if self.source else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "Choose audio or video", start, FILE_FILTER)
        if path:
            self._set_source(Path(path))

    def _set_source(self, path: Path):
        self.source = path
        self.drop.title.setText(path.name)
        duration = audio.probe_duration(path)
        parts = [f"Duration {hms(duration)}" if duration else "Duration unknown",
                 f"{path.stat().st_size / 2**20:.0f} MB"]
        resume = self._resume_hint(path)
        if resume:
            parts.append(resume)
        self.drop.detail.setText(" · ".join(parts))
        self._load_existing_outputs(path)
        self._set_busy(False)

    def _resume_hint(self, path: Path) -> str:
        manifest = read_json(Job.root_for(path, Path(self.settings.output_dir)) / "manifest.json")
        if not manifest or not manifest.get("chunks"):
            return ""
        stages = manifest.get("stages", {})
        if stages.get("summarized") or (stages.get("transcribed") and not self.settings.summarize):
            return "Already processed: results loaded below"
        root = Job.root_for(path, Path(self.settings.output_dir))
        done = sum(1 for i in range(len(manifest["chunks"])) if (root / "parts" / f"{i:04d}.json").exists())
        return f"Earlier progress found: {done}/{len(manifest['chunks'])} parts done, Start will resume"

    def _load_existing_outputs(self, path: Path):
        root = Job.root_for(path, Path(self.settings.output_dir))
        self.job_dir = root if root.exists() else None
        self._show_outputs(root)

    # ------------------------------------------------------------------ run control

    def _start(self):
        if not self.source:
            self._choose_file()
            if not self.source:
                return
        if self.settings.summarize and not self.settings.llm_path:
            choice = QMessageBox.question(
                self, "No summary model",
                "No summary model is selected, so only the transcript will be created.\n\n"
                "Choose a GGUF model in Settings first?",
                QMessageBox.Yes | QMessageBox.No)
            if choice == QMessageBox.Yes:
                self._open_settings()
                return
        self.transcript.clear()
        self.chapters.clear()
        self.notes.clear()
        self.log.clear()
        self.progress.setValue(0)
        self.status.setText("Starting…")
        self.started_at = time.monotonic()
        self.runner.start(str(self.source), self.settings.to_dict())
        self._set_busy(True)

    def _cancel(self):
        self.runner.cancel()
        self.cancel_btn.setEnabled(False)
        self.status.setText("Stopping after the current step…")

    def _set_busy(self, busy: bool):
        self.start_btn.setEnabled(not busy and self.source is not None)
        self.open_btn.setEnabled(not busy)
        self.settings_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)
        self.folder_btn.setEnabled(self.job_dir is not None)

    def _on_event(self, ev: dict):
        kind = ev.get("type")
        if kind == "progress":
            self.progress.setValue(int(ev["fraction"] * 1000))
            eta = f"  ·  about {hms(ev['eta'])} left" if ev.get("eta") else ""
            self.status.setText(f"{ev['message']}{eta}")
        elif kind == "segments":
            for seg in ev["items"]:
                self.transcript.appendPlainText(f"[{hms(seg['start'])}] {seg['text']}")
        elif kind == "log":
            prefix = "WARNING: " if ev["level"] == "warning" else ""
            self.log.appendPlainText(f"{prefix}{ev['message']}")
            if ev["level"] == "warning":
                self.status.setText(ev["message"])
        elif kind == "job":
            self.job_dir = Path(ev["job_dir"])
            self.folder_btn.setEnabled(True)
        elif kind == "done":
            self.progress.setValue(1000)
            took = hms(time.monotonic() - self.started_at) if self.started_at else ""
            self.status.setText(f"Done in {took}. Files are in the output folder.")
            self._show_outputs(Path(ev["job_dir"]))
            if (Path(ev["job_dir"]) / "chapters.md").exists():
                self.tabs.setCurrentWidget(self.chapters)
        elif kind == "cancelled":
            self.status.setText("Stopped. Finished parts are saved; press Start to resume.")
        elif kind == "error":
            self.log.appendPlainText("ERROR: " + ev["message"])
            if ev.get("traceback"):
                self.log.appendPlainText(ev["traceback"])
            self.status.setText("Failed: " + ev["message"])
            QMessageBox.critical(self, "Processing failed",
                                 ev["message"] + "\n\nDetails are in the Log tab.")

    def _on_finished(self):
        self._set_busy(False)

    def _show_outputs(self, root: Path):
        transcript = root / "transcript.txt"
        if transcript.exists():
            self.transcript.setPlainText(transcript.read_text(encoding="utf-8"))
        else:
            self.transcript.clear()
        for name, view in (("chapters.md", self.chapters), ("notes.md", self.notes)):
            path = root / name
            if path.exists():
                view.setMarkdown(path.read_text(encoding="utf-8"))
            else:
                view.clear()

    # ------------------------------------------------------------------ settings & misc

    def _open_settings(self):
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec():
            self.settings = dialog.result_settings()
            save_settings(self.settings)
            self._refresh_model_label()
            if self.source:
                self._set_source(self.source)

    def _refresh_model_label(self):
        llm = Path(self.settings.llm_path).stem if self.settings.llm_path else "none"
        summary = llm if self.settings.summarize else "off"
        self.model_label.setText(f"Whisper: {self.settings.whisper_model}  ·  Summary: {summary}")

    def _open_folder(self):
        if self.job_dir and Path(self.job_dir).exists():
            os.startfile(self.job_dir)

    def closeEvent(self, event):
        if self.runner.is_running():
            choice = QMessageBox.question(
                self, "Job running",
                "A job is still running. Stop it and quit?\n\nFinished parts are saved and will resume next time.",
                QMessageBox.Yes | QMessageBox.No)
            if choice != QMessageBox.Yes:
                event.ignore()
                return
            self.runner.shutdown()
        event.accept()
