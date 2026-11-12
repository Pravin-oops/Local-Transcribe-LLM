import os
import time
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QTabWidget, QTextBrowser, QVBoxLayout,
                               QWidget)

from localscribe import audio, models, subtitles, translator, youtube
from localscribe.config import OUTPUTS, load_settings, models_dir, save_settings
from localscribe.estimates import reading_summary
from localscribe.jobs import Job, read_json
from localscribe.timefmt import hms
from localscribe.ui import theme
from localscribe.ui.options_panel import OptionsPanel
from localscribe.ui.settings_dialog import SettingsDialog
from localscribe.ui.widgets import DropZone
from localscribe.ui.worker import JobRunner, ModelDownload, YouTubeProbe

FILE_FILTER = "Audio, video and subtitles ({});;Subtitles ({});;All files (*)".format(
    " ".join(f"*{ext}" for ext in audio.AUDIO_EXTENSIONS + subtitles.SUBTITLE_EXTENSIONS),
    " ".join(f"*{ext}" for ext in subtitles.SUBTITLE_EXTENSIONS))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LocalScribe")
        self.resize(1080, 940)
        self.setMinimumSize(900, 760)
        self.setAcceptDrops(True)
        self.settings = load_settings()
        self.source = None     # a local file ...
        self.url = None        # ... or a YouTube link (then self.video is what the lookup found)
        self.video = None
        self.probe = None
        self.duration = None
        self.job_dir = None
        self.started_at = None
        self.download = None

        self.runner = JobRunner(self)
        self.runner.event.connect(self._on_event)
        self.runner.finished.connect(self._on_finished)

        # Header
        title = QLabel("LocalScribe")
        title.setObjectName("Title")
        tagline = QLabel("Private, offline transcripts, notes and stories")
        tagline.setObjectName("Muted")
        self.settings_btn = QPushButton(theme.ICON_SETTINGS)
        self.settings_btn.setObjectName("IconButton")
        self.settings_btn.setToolTip("Settings")
        self.settings_btn.setCursor(Qt.PointingHandCursor)
        self.settings_btn.clicked.connect(self._open_settings)
        header = QHBoxLayout()
        header.addWidget(title, 0, Qt.AlignVCenter)
        header.addSpacing(10)
        header.addWidget(tagline, 0, Qt.AlignVCenter)
        header.addStretch(1)
        header.addWidget(self.settings_btn)

        # Drop zone and options
        self.drop = DropZone()
        self.drop.clicked.connect(self._choose_file)
        self.link = QLineEdit()
        self.link.setPlaceholderText("Or paste a YouTube link: its captions are summarized, no transcription needed")
        self.link.setClearButtonEnabled(True)
        self.link.textChanged.connect(self._link_edited)
        self.link.returnPressed.connect(self._open_link)
        self.link_btn = QPushButton("Check captions")
        self.link_btn.setCursor(Qt.PointingHandCursor)
        self.link_btn.clicked.connect(self._open_link)
        link_row = QHBoxLayout()
        link_row.setSpacing(10)
        link_row.addWidget(self.link, 1)
        link_row.addWidget(self.link_btn)
        self.panel = OptionsPanel(self.settings)
        self.panel.changed.connect(self._outputs_changed)
        self.panel.download_whisper.connect(self._download_whisper)
        self.panel.add_llm.connect(self._add_llm)

        # Actions and progress
        self.start_btn = QPushButton("Start")
        self.start_btn.setObjectName("Primary")
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.cancel_btn = QPushButton("Cancel")
        self.folder_btn = QPushButton(f"{theme.ICON_FOLDER}")
        self.folder_btn.setObjectName("IconButton")
        self.folder_btn.setToolTip("Open the output folder")
        self.start_btn.clicked.connect(self._start)
        self.cancel_btn.clicked.connect(self._cancel)
        self.folder_btn.clicked.connect(self._open_folder)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.status = QLabel("Choose a file to begin")
        self.status.setObjectName("Muted")
        actions = QHBoxLayout()
        actions.setSpacing(10)
        actions.addWidget(self.start_btn)
        actions.addWidget(self.cancel_btn)
        actions.addSpacing(8)
        actions.addWidget(self.status, 1)
        actions.addWidget(self.folder_btn)

        # Results
        mono = QFont("Cascadia Mono")
        mono.setStyleHint(QFont.Monospace)
        self.transcript = QPlainTextEdit(readOnly=True)
        self.transcript.setFont(mono)
        # One rendered-Markdown view per LLM output, shown when selected or present.
        self.views = {key: QTextBrowser(openExternalLinks=True) for key in OUTPUTS}
        self.log = QPlainTextEdit(readOnly=True)
        self.log.setFont(mono)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(False)
        self.tabs.addTab(self.transcript, "Transcript")
        for key, label in OUTPUTS.items():
            self.tabs.addTab(self.views[key], label)
        self.translated = QTextBrowser(openExternalLinks=True)  # the story in the selected language
        self.tabs.addTab(self.translated, "Translated story")
        self.tabs.addTab(self.log, "Log")

        page = QWidget()
        page.setObjectName("Page")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(26, 20, 26, 22)
        layout.setSpacing(14)
        layout.addLayout(header)
        layout.addWidget(self.drop)
        layout.addLayout(link_row)
        layout.addWidget(self.panel)
        layout.addLayout(actions)
        layout.addWidget(self.progress)
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(page)

        self._refresh_tabs()
        self._update_reading()
        self._set_busy(False)

    # ------------------------------------------------------------------ file selection

    def dragEnterEvent(self, event):
        if not self.runner.is_running() and (self._dropped_file(event) or self._dropped_link(event)):
            event.acceptProposedAction()
            self.drop.set_active(True)

    def dragLeaveEvent(self, event):
        self.drop.set_active(False)

    def dropEvent(self, event):
        self.drop.set_active(False)
        path, link = self._dropped_file(event), self._dropped_link(event)
        if path:
            self._open_source(path)
        elif link:
            self.link.setText(link)  # opens it through _link_edited

    @staticmethod
    def _dropped_link(event):
        """A YouTube link dragged from a browser's address bar or page."""
        mime = event.mimeData()
        texts = [u.toString() for u in mime.urls()] + ([mime.text()] if mime.hasText() else [])
        return next((t.strip() for t in texts if youtube.is_url(t)), None)

    @staticmethod
    def _dropped_file(event):
        urls = event.mimeData().urls()
        if len(urls) == 1 and urls[0].isLocalFile():
            path = Path(urls[0].toLocalFile())
            if path.is_file():
                return path
        return None

    def _choose_file(self):
        if self.runner.is_running():
            return
        start = str(self.source.parent) if self.source else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "Choose audio, video or subtitles", start, FILE_FILTER)
        if path:
            self._open_source(Path(path))

    def _open_source(self, path: Path):
        """A file the user just picked; subtitles default to the Movie / Engaging Story preset."""
        if subtitles.is_subtitle(path) and self.settings.content_type != "movie":
            self.settings.apply_preset("movie")
            self.panel.load(self.settings)
            save_settings(self.settings)
        self.url = self.video = None
        self.link.blockSignals(True)
        self.link.clear()
        self.link.blockSignals(False)
        self._set_source(path)

    # ------------------------------------------------------------------ YouTube link

    def _link_edited(self, text: str):
        """Open a pasted link straight away; typing still needs Enter or the button."""
        vid = youtube.video_id(text)
        if vid and youtube.watch_url(vid) != self.url:
            self._open_link()

    def _open_link(self):
        """Look the video up: are there captions to summarize, or must the audio be transcribed?"""
        if self.runner.is_running():
            return
        vid = youtube.video_id(self.link.text())
        if not vid:
            self.status.setText("That does not look like a YouTube video link")
            return
        if "summary" not in self.settings.outputs:  # a link is there to be summarized
            self.settings.apply_preset("lecture")
            self.panel.load(self.settings)
            save_settings(self.settings)
            self._refresh_tabs()
        self.source, self.url, self.video = None, youtube.watch_url(vid), None
        self._show_video()
        self.status.setText("Checking YouTube for captions…")
        self.probe = YouTubeProbe(self.url, self.settings.language, self)
        self.probe.done.connect(self._probe_done)
        self.probe.failed.connect(self._probe_failed)
        self.probe.finished.connect(self.probe.deleteLater)
        self.probe.start()

    def _probe_done(self, url: str, info: dict):
        if url != self.url:
            return  # an answer for a link that has since been replaced
        self.video = info
        self._show_video()
        self.status.setText("Captions found: ready to summarize" if info["captions"]
                            else "No captions: Start will download the audio and transcribe it")

    def _probe_failed(self, url: str, error: str):
        if url != self.url:
            return
        self.video = {"error": error}
        self._show_video()
        self.status.setText("Could not check this link")

    def _show_video(self):
        """Describe the linked video in the drop zone: what was found and what Start will do."""
        saved = youtube.cached(youtube.video_id(self.url), youtube.downloads_dir(self.settings.output_dir))
        info = self.video or {}
        self.duration = info.get("duration")
        parts = ["YouTube"] + ([hms(self.duration)] if self.duration else [])
        captions = saved.suffix.lower() == ".vtt" if saved else bool(info.get("captions"))
        if saved:
            parts.append("Captions saved earlier · transcription skipped" if captions
                         else "Audio saved earlier: it will be transcribed")
        elif captions:
            parts.append(f"{youtube.CAPTION_KINDS[info['captions']].capitalize()} captions found"
                         " · transcription skipped")
        elif "error" in info:
            parts.append(f"Could not check this link: {info['error']}")
        elif self.video is None:
            parts.append("Checking for captions…")
        else:
            parts.append("No captions: the audio will be downloaded and transcribed")
        self.panel.set_audio_available(not captions)
        resume = self._resume_hint(saved) if saved else ""
        if resume:
            parts.append(resume)
        title = info.get("title") or (saved.stem.rsplit(" [", 1)[0] if saved else "YouTube video")
        self.drop.set_file(title, " · ".join(parts))
        if saved:
            self._load_existing_outputs(saved)
        else:
            self.job_dir = None
            self._show_outputs(Path(self.settings.output_dir) / "jobs" / "none")
        self._update_reading()
        self._set_busy(False)

    def _refresh_source(self):
        """Re-describe the current file or link after options or settings changed."""
        if self.url:
            self._show_video()
        elif self.source:
            self._set_source(self.source)
        else:
            self._update_reading()

    def _set_source(self, path: Path):
        self.source = path
        is_subs = subtitles.is_subtitle(path)
        if is_subs:  # the subtitles are the transcript: no audio for Whisper or speaker detection
            segs = None
            try:
                segs = subtitles.load(path)
            except (OSError, subtitles.SubtitleError):
                pass
            self.duration = segs[-1]["end"] if segs else None
            parts = ["Subtitles", subtitles.describe(path), "transcription skipped"]
        else:
            self.duration = audio.probe_duration(path)
            parts = [hms(self.duration) if self.duration else "Duration unknown",
                     f"{path.stat().st_size / 2**20:.0f} MB"]
        self.panel.set_audio_available(not is_subs)
        resume = self._resume_hint(path)
        if resume:
            parts.append(resume)
        self.drop.set_file(path.name, " · ".join(parts))
        self._load_existing_outputs(path)
        self.status.setText("Ready to start")
        self._update_reading()
        self._set_busy(False)

    def _resume_hint(self, path: Path) -> str:
        root = Job.root_for(path, Path(self.settings.output_dir))
        manifest = read_json(root / "manifest.json")
        if not manifest:
            return ""
        if manifest.get("stages", {}).get("transcribed"):
            done = manifest.get("outputs_done", {})
            missing = [OUTPUTS[o] for o in self.settings.outputs if o not in done]
            code = self.settings.translate_to
            if code in translator.LANGUAGES and "story" in self.settings.outputs and f"story.{code}" not in done:
                missing.append(f"Story in {translator.language_name(code)}")
            if not missing:
                return "Already processed: results loaded below"
            return f"Transcript ready: Start will add {', '.join(missing)}"
        if not manifest.get("chunks"):
            return ""
        parts = sum(1 for i in range(len(manifest["chunks"])) if (root / "parts" / f"{i:04d}.json").exists())
        return f"Earlier progress found: {parts}/{len(manifest['chunks'])} parts done, Start will resume"

    def _load_existing_outputs(self, path: Path):
        root = Job.root_for(path, Path(self.settings.output_dir))
        self.job_dir = root if root.exists() else None
        self._show_outputs(root)

    # ------------------------------------------------------------------ run control

    def _start(self):
        if not self.source and not self.url:
            self._choose_file()
            if not self.source:
                return
        if self.settings.outputs and not self.settings.llm_path:
            choice = QMessageBox.question(
                self, "No LLM model",
                "No LLM model is selected, so only the transcript will be created.\n\n"
                "Add a .gguf model first?",
                QMessageBox.Yes | QMessageBox.No)
            if choice == QMessageBox.Yes:
                self._add_llm()
                return
        self.transcript.clear()
        for view in [*self.views.values(), self.translated]:
            view.clear()
        self.log.clear()
        self.progress.setValue(0)
        self.status.setText("Starting…")
        self.started_at = time.monotonic()
        self.runner.start(self.url or str(self.source), self.settings.to_dict())
        self._set_busy(True)

    def _cancel(self):
        self.runner.cancel()
        self.cancel_btn.setEnabled(False)
        self.status.setText("Stopping after the current step…")

    def _set_busy(self, busy: bool):
        self.start_btn.setEnabled(not busy and (self.source is not None or self.url is not None))
        self.drop.setEnabled(not busy)
        self.link.setEnabled(not busy)
        self.link_btn.setEnabled(not busy)
        self.settings_btn.setEnabled(not busy)
        self.panel.setEnabled(not busy)
        self.cancel_btn.setVisible(busy)
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
            self.status.setText(f"Done in {took}")
            self._show_outputs(Path(ev["job_dir"]))
            self._update_reading()
            if self.tabs.isTabVisible(self.tabs.indexOf(self.translated)) and self.translated.toPlainText():
                self.tabs.setCurrentWidget(self.translated)  # the translation was asked for: show it
            else:
                for key in ("story", "summary", "notes"):  # most specific output first
                    if key in self.settings.outputs and (Path(ev["job_dir"]) / f"{key}.md").exists():
                        self.tabs.setCurrentWidget(self.views[key])
                        break
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
        if self.url:
            self._show_video()  # the captions or audio are saved now
        self._set_busy(False)

    def _show_outputs(self, root: Path):
        transcript = root / "transcript.txt"
        if transcript.exists():
            self.transcript.setPlainText(transcript.read_text(encoding="utf-8"))
        else:
            self.transcript.clear()
        for key, view in self.views.items():
            path = root / f"{key}.md"
            if path.exists():
                view.setMarkdown(path.read_text(encoding="utf-8"))
            else:
                view.clear()
        path = self._translation_path(root)
        if path:
            self.translated.setMarkdown(path.read_text(encoding="utf-8"))
        else:
            self.translated.clear()
        self._refresh_tabs()

    def _translation_path(self, root):
        """The translated story in ``root``: the selected language first, else any that exists."""
        if root is None:
            return None
        codes = dict.fromkeys([self.settings.translate_to, *translator.LANGUAGES])
        return next((p for code in codes if code in translator.LANGUAGES
                     and (p := Path(root) / translator.output_name(code)).exists()), None)

    def _refresh_tabs(self):
        """Show an output's tab when it is selected or already exists for this file."""
        for key, view in self.views.items():
            exists = self.job_dir is not None and (Path(self.job_dir) / f"{key}.md").exists()
            self.tabs.setTabVisible(self.tabs.indexOf(view), key in self.settings.outputs or exists)
        path = self._translation_path(self.job_dir)
        code = path.name.split(".")[1] if path else self.settings.translate_to
        wanted = self.settings.translate_to in translator.LANGUAGES and "story" in self.settings.outputs
        index = self.tabs.indexOf(self.translated)
        self.tabs.setTabVisible(index, wanted or path is not None)
        self.tabs.setTabText(index, f"Story in {translator.language_name(code) or 'translation'}")

    def _update_reading(self):
        self.panel.set_reading(reading_summary(self.settings.outputs, self.duration,
                                               self.settings.story_minutes, self.job_dir))

    # ------------------------------------------------------------------ models

    def _download_whisper(self, name: str):
        mb = models.WHISPER_CATALOG[name][1]
        choice = QMessageBox.question(
            self, "Download Whisper model",
            f"Download '{name}' ({models.pretty_size(mb * 1_000_000)}) from Hugging Face?\n\n"
            "This happens once; the model then works offline.")
        if choice != QMessageBox.Yes:
            return
        self.panel.whisper.set_busy(f"Downloading {name}…")
        self.status.setText(f"Downloading Whisper model {name}…")
        self.download = ModelDownload(name, self)
        self.download.done.connect(self._download_done)
        self.download.failed.connect(self._download_failed)
        self.download.start()

    def _download_done(self, name: str):
        self.settings.whisper_model = name
        save_settings(self.settings)
        self.panel.refresh_models(self.settings)
        self.status.setText(f"Whisper model {name} is ready")

    def _download_failed(self, name: str, error: str):
        self.panel.refresh_models(self.settings)
        self.status.setText(f"Download of {name} failed")
        QMessageBox.warning(self, "Download failed", f"Could not download '{name}':\n\n{error}")

    def _add_llm(self):
        path, _ = QFileDialog.getOpenFileName(self, "Add an LLM model", str(models_dir()), "GGUF models (*.gguf)")
        if path:
            self.settings.llm_path = path
            save_settings(self.settings)
            self.panel.refresh_models(self.settings)

    # ------------------------------------------------------------------ settings & misc

    def _outputs_changed(self):
        self.panel.apply_to(self.settings)
        save_settings(self.settings)
        self._refresh_tabs()
        self._refresh_source()

    def _open_settings(self):
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec():
            self.settings = dialog.result_settings()
            save_settings(self.settings)
            self.panel.refresh_models(self.settings)
            self._refresh_source()

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
        for probe in self.findChildren(YouTubeProbe):  # a lookup still on the network must not outlive the window
            if probe.isRunning():
                probe.terminate()
                probe.wait(1000)
        event.accept()
