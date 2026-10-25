"""faster-whisper wrapper that transcribes one chunk at a time."""
import gc
import re

SAMPLE_RATE = 16_000


class Cancelled(Exception):
    """Raised when the user cancels; finished chunks stay on disk."""


def _norm(text: str) -> str:
    return re.sub(r"[^\w ]+", "", text.lower()).strip()


def keep_segment(start: float, end: float, text: str, chunk_length: float, prev_text: str) -> bool:
    """Filter the two hallucination patterns Whisper shows on trailing silence.

    - A segment near the chunk end whose timestamps run past the audio it was
      given (e.g. "Thank you." stamped 90.2-100.5 s in a 90.4 s chunk).
    - An exact repeat of the previous segment's text.
    """
    if not _norm(text):
        return False
    if start >= chunk_length - 1.5 and end > chunk_length + 0.5:
        return False
    return _norm(text) != _norm(prev_text or "")


class Transcriber:
    def __init__(self, model_name: str, device: str, compute_type: str, cpu_threads: int = 0,
                 batch_size: int = 0, on_status=None):
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        kwargs = dict(device=device, compute_type=compute_type, cpu_threads=cpu_threads)
        try:
            # Prefer the cached copy so no network call happens once downloaded.
            self.model = WhisperModel(model_name, local_files_only=True, **kwargs)
        except Exception:
            if on_status:
                on_status(f"Downloading Whisper model '{model_name}' (first run only)...")
            self.model = WhisperModel(model_name, **kwargs)
        self.batch_size = batch_size
        self.pipeline = BatchedInferencePipeline(self.model) if batch_size > 1 else None
        self.device = device
        self.compute_type = compute_type

    def transcribe(self, audio, offset: float, language: str, beam_size: int,
                   initial_prompt: str = None, on_segment=None, cancel=None) -> list:
        """Transcribe a float32 16 kHz chunk; timestamps are shifted by ``offset``."""
        options = dict(language=language or None, beam_size=beam_size, vad_filter=True,
                       initial_prompt=initial_prompt or None)
        if self.pipeline is not None:
            segments, _ = self.pipeline.transcribe(audio, batch_size=self.batch_size, **options)
        else:
            segments, _ = self.model.transcribe(audio, **options)

        length = len(audio) / SAMPLE_RATE
        out = []
        for seg in segments:  # lazy generator: decoding happens while iterating
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            text = seg.text.strip()
            if not keep_segment(seg.start, seg.end, text, length, out[-1]["text"] if out else None):
                continue
            item = {"start": round(seg.start + offset, 2),
                    "end": round(min(seg.end, length) + offset, 2), "text": text}
            out.append(item)
            if on_segment:
                on_segment(item)
        return out

    def close(self) -> None:
        """Release VRAM so the summarizer can use it."""
        try:
            self.model.model.unload_model()
        except Exception:
            pass
        self.pipeline = None
        self.model = None
        gc.collect()
