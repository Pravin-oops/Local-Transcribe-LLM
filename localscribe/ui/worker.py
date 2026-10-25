"""Runs the pipeline in a separate process and relays its events as Qt signals.

A separate process (not a thread) keeps the window responsive during heavy
native work, lets Cancel force-stop a stuck model, and means a crash inside
CUDA/llama.cpp cannot take the window down. Finished parts are on disk, so a
stopped job resumes on the next Start.
"""
import multiprocessing as mp
import queue

from PySide6.QtCore import QObject, QTimer, Signal

from localscribe.pipeline import process_main

TERMINAL = ("done", "error", "cancelled")
FORCE_STOP_MS = 20_000


class JobRunner(QObject):
    event = Signal(dict)
    finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ctx = mp.get_context("spawn")
        self._proc = self._queue = self._cancel = None
        self._got_terminal = False
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._poll)

    def is_running(self) -> bool:
        return self._proc is not None

    def start(self, source: str, settings: dict, resummarize: bool = False) -> None:
        if self.is_running():
            return
        self._queue = self._ctx.Queue()
        self._cancel = self._ctx.Event()
        self._got_terminal = False
        self._proc = self._ctx.Process(target=process_main, daemon=True,
                                       args=(source, settings, self._queue, self._cancel, resummarize))
        self._proc.start()
        self._timer.start()

    def cancel(self) -> None:
        """Ask the worker to stop after its current step; force it if it hangs."""
        if self._cancel is not None:
            self._cancel.set()
            QTimer.singleShot(FORCE_STOP_MS, self._force_stop)

    def shutdown(self) -> None:
        """Stop immediately (window closing)."""
        if self._proc is None:
            return
        self._cancel.set()
        self._proc.join(3)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(2)
        self._timer.stop()
        self._proc = None

    def _force_stop(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()

    def _drain(self, limit: int) -> None:
        for _ in range(limit):
            try:
                ev = self._queue.get_nowait()
            except (queue.Empty, OSError, EOFError):
                return
            if ev.get("type") in TERMINAL:
                self._got_terminal = True
            self.event.emit(ev)

    def _poll(self) -> None:
        if self._proc is None:
            self._timer.stop()
            return
        self._drain(500)  # bounded so a flood of segments cannot freeze the UI
        if self._proc.is_alive():
            return
        self._drain(100_000)
        if not self._got_terminal:
            code = self._proc.exitcode
            if self._cancel.is_set():
                self.event.emit({"type": "cancelled"})
            else:
                self.event.emit({"type": "error",
                                 "message": f"The worker stopped unexpectedly (exit code {code}). "
                                            "Finished parts are saved; press Start to resume."})
        self._timer.stop()
        self._proc = None
        self.finished.emit()
