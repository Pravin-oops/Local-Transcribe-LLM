"""Run one file through prepare -> transcribe -> summarize, resuming finished work.

Everything reports through ``emit(event: dict)`` so the same code drives the
CLI and the desktop UI (where events cross a multiprocessing queue). Events:

    {"type": "job", "job_dir"}
    {"type": "progress", "stage", "fraction", "message", "eta"}
    {"type": "segments", "items": [{"start", "end", "text"}, ...]}
    {"type": "log", "level", "message"}
    {"type": "done", "job_dir", "files"}
    {"type": "cancelled"} / {"type": "error", "message", "traceback"}
"""
import time
import traceback
from datetime import datetime
from pathlib import Path

from localscribe import audio, hardware
from localscribe.config import Settings
from localscribe.jobs import Job
from localscribe.merger import write_transcripts
from localscribe.summarizer import LlamaBackend, auto_gpu_layers, render_chapters, render_notes, summarize
from localscribe.timefmt import hms
from localscribe.transcriber import Cancelled, Transcriber

SR = audio.SAMPLE_RATE


class _NeverCancel:
    @staticmethod
    def is_set() -> bool:
        return False


class _Reporter:
    """Maps per-stage progress onto one 0..1 bar and mirrors logs to the job folder."""

    def __init__(self, emit, summarizing: bool):
        self.emit = emit
        self.log_path = None
        split = 0.80 if summarizing else 1.0
        self.spans = {"prepare": (0.0, 0.05), "transcribe": (0.05, split), "summarize": (split, 1.0)}

    def progress(self, stage: str, fraction: float, message: str, eta: float = None) -> None:
        lo, hi = self.spans[stage]
        fraction = lo + (hi - lo) * max(0.0, min(1.0, fraction))
        self.emit({"type": "progress", "stage": stage, "fraction": fraction,
                   "message": message, "eta": eta})

    def log(self, message: str, level: str = "info") -> None:
        if self.log_path:
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"{stamp} [{level}] {message}\n")
        self.emit({"type": "log", "level": level, "message": message})

    def warn(self, message: str) -> None:
        self.log(message, "warning")


def _default_transcriber(settings: Settings, device: str, compute_type: str, on_status):
    cores = hardware.detect().physical_cores
    return Transcriber(settings.whisper_model, device, compute_type, cpu_threads=cores,
                       batch_size=settings.batch_size, on_status=on_status)


def _default_backend(settings: Settings, on_status):
    hw = hardware.detect()
    layers = settings.llm_gpu_layers
    if layers < 0:
        layers = auto_gpu_layers(settings.llm_path, settings.llm_ctx, hardware.vram_free_mb()) \
            if hw.has_cuda else 0
    return LlamaBackend(settings.llm_path, settings.llm_ctx, layers, hw.physical_cores, on_status)


def run_job(source, settings: Settings, emit, cancel=None, resummarize: bool = False,
            transcriber_factory=None, backend_factory=None) -> dict:
    cancel = cancel or _NeverCancel()
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"Audio file not found: {source}")
    summarizing = settings.summarize and bool(settings.llm_path)
    report = _Reporter(emit, summarizing)
    job = Job.for_source(source, Path(settings.output_dir))
    report.log_path = job.output("log.txt")
    emit({"type": "job", "job_dir": str(job.root)})
    report.log(f"Job started for {source}")

    _prepare(job, source, settings, report, cancel)
    segments = _transcribe(job, settings, report, emit, cancel,
                           transcriber_factory or _default_transcriber)
    files = write_transcripts(segments, job.root)
    job.set_stage("transcribed", True)
    report.log(f"Transcript written: {len(segments)} segments")

    if settings.summarize and not settings.llm_path:
        report.warn("Summary skipped: no summary model selected in Settings.")
    elif summarizing:
        files.update(_summarize(job, source, segments, settings, report, cancel, resummarize,
                                backend_factory or _default_backend))

    result = {"type": "done", "job_dir": str(job.root), "files": files}
    report.log("Job finished")
    emit(result)
    return result


def _prepare(job: Job, source: Path, settings: Settings, report: _Reporter, cancel) -> None:
    manifest = job.load_manifest()
    if (job.pcm_path.exists() and manifest.get("chunks")
            and manifest.get("chunk_minutes") == settings.chunk_minutes):
        report.progress("prepare", 1.0, "Audio already prepared")
        return
    if not job.pcm_path.exists():
        report.progress("prepare", 0.0, f"Decoding {source.name}")
        audio.convert_to_pcm(source, job.pcm_path, cancel=cancel,
                             on_progress=lambda f: report.progress("prepare", 0.9 * f, "Decoding audio"))
        if cancel.is_set():
            raise Cancelled()
    samples = audio.open_pcm(job.pcm_path)
    chunks = audio.plan_chunks(samples, settings.chunk_minutes * 60)
    duration = len(samples) / SR
    del samples  # release the memory map (Windows keeps the file locked while mapped)
    if chunks != manifest.get("chunks"):
        job.clear_dir(job.parts_dir)
        job.clear_dir(job.notes_dir)
    job.update_manifest(duration=duration, chunks=chunks, chunk_minutes=settings.chunk_minutes)
    report.log(f"Audio {hms(duration)} split into {len(chunks)} parts")
    report.progress("prepare", 1.0, f"Prepared {hms(duration)} of audio in {len(chunks)} parts")


def _transcribe(job: Job, settings: Settings, report: _Reporter, emit, cancel, factory) -> list:
    manifest = job.load_manifest()
    chunks = manifest["chunks"]
    config = {"model": settings.whisper_model, "language": settings.language,
              "beam_size": settings.beam_size, "batch_size": settings.batch_size}
    if manifest.get("transcribe_config") != config:
        job.clear_dir(job.parts_dir)
        job.clear_dir(job.notes_dir)
        job.set_stage("transcribed", False)
        job.set_stage("summarized", False)
        job.update_manifest(transcribe_config=config)

    total_s = manifest["duration"] or 1.0
    by_chunk = {}
    for i in range(len(chunks)):
        part = job.read_part(i)
        if part is not None:
            by_chunk[i] = part
            emit({"type": "segments", "items": part})
    remaining = [i for i in range(len(chunks)) if i not in by_chunk]
    done_s = sum((chunks[i][1] - chunks[i][0]) / SR for i in by_chunk)
    if by_chunk and remaining:
        report.log(f"Resuming: {len(by_chunk)}/{len(chunks)} parts already transcribed")
    if not remaining:
        report.progress("transcribe", 1.0, "Transcript already complete")
        return [seg for i in range(len(chunks)) for seg in by_chunk[i]]

    report.progress("transcribe", done_s / total_s, f"Loading Whisper model '{settings.whisper_model}'")
    device, compute = hardware.whisper_runtime(settings.device, settings.compute_type)
    try:
        transcriber = factory(settings, device, compute, report.log)
    except Exception as exc:
        if device != "cuda":
            raise
        report.warn(f"GPU unavailable for Whisper ({exc}); falling back to CPU")
        transcriber = factory(settings, "cpu", "int8", report.log)
    report.log(f"Whisper '{settings.whisper_model}' on {transcriber.device} ({transcriber.compute_type})")

    samples = audio.open_pcm(job.pcm_path)
    session_s, t0 = 0.0, time.monotonic()
    try:
        for i in remaining:
            if cancel.is_set():
                raise Cancelled()
            start, end = chunks[i]
            offset, length = start / SR, (end - start) / SR
            prev = by_chunk.get(i - 1)
            prompt = " ".join(s["text"] for s in prev[-3:])[-400:] if prev else None
            label = f"Transcribing part {i + 1}/{len(chunks)}"
            report.progress("transcribe", done_s / total_s, label)

            def on_segment(seg, done_s=done_s, session_s=session_s, offset=offset, label=label):
                emit({"type": "segments", "items": [seg]})
                within = max(0.0, seg["end"] - offset)
                elapsed = time.monotonic() - t0
                rate = (session_s + within) / elapsed if elapsed > 1 else 0
                eta = (total_s - done_s - within) / rate if rate > 0 else None
                report.progress("transcribe", (done_s + within) / total_s, label, eta)

            chunk_audio = audio.to_float(samples[start:end])
            try:
                segs = transcriber.transcribe(chunk_audio, offset, settings.language,
                                              settings.beam_size, prompt, on_segment, cancel)
            except Cancelled:
                raise
            except Exception as exc:
                if transcriber.device != "cuda":
                    raise
                report.warn(f"GPU transcription failed ({exc}); retrying this part on CPU")
                transcriber.close()
                transcriber = factory(settings, "cpu", "int8", report.log)
                segs = transcriber.transcribe(chunk_audio, offset, settings.language,
                                              settings.beam_size, prompt, on_segment, cancel)
            job.write_part(i, segs)
            by_chunk[i] = segs
            done_s += length
            session_s += length
    finally:
        transcriber.close()
        del samples
    report.log(f"Transcribed {hms(session_s)} of audio in {hms(time.monotonic() - t0)}")
    return [seg for i in range(len(chunks)) for seg in by_chunk[i]]


def _summarize(job: Job, source: Path, segments: list, settings: Settings, report: _Reporter,
               cancel, resummarize: bool, factory) -> dict:
    llm_path = Path(settings.llm_path)
    if not llm_path.is_file():
        report.warn(f"Summary skipped: model file not found: {llm_path}")
        return {}
    files = {name: str(job.output(name)) for name in ("chapters.md", "notes.md", "summary.md")}
    manifest = job.load_manifest()
    if (not resummarize and job.stage_done("summarized")
            and manifest.get("summary_model") == llm_path.stem
            and all(Path(p).exists() for p in files.values())):
        report.progress("summarize", 1.0, "Summary already complete")
        return files
    if resummarize:
        job.clear_dir(job.notes_dir)

    report.progress("summarize", 0.0, f"Loading summary model {llm_path.stem}")
    backend = factory(settings, report.log)
    report.log(f"Summary model {backend.name}: context {backend.n_ctx}, "
               f"GPU layers {getattr(backend, 'gpu_layers', '?')}")
    t0 = time.monotonic()
    try:
        result = summarize(segments, backend, job, source.stem,
                           lambda f, m: report.progress("summarize", f, m), cancel, report.warn)
    finally:
        backend.close()

    duration = job.load_manifest().get("duration", 0)
    # *...* rather than _..._: model names like Q4_K_M contain underscores.
    header = (f"# {source.stem}\n\n*Duration {hms(duration)} · Transcribed with "
              f"{settings.whisper_model} · Summarized with {backend.name}*\n\n")
    chapters_md = render_chapters(result["chapters"])
    notes_md = render_notes(result["notes"])
    job.output("chapters.md").write_text(header + chapters_md, encoding="utf-8")
    job.output("notes.md").write_text(header + notes_md, encoding="utf-8")
    job.output("summary.md").write_text(header + chapters_md + "\n" + notes_md, encoding="utf-8")
    job.update_manifest(summary_model=backend.name)
    job.set_stage("summarized", True)
    report.log(f"Summary written in {hms(time.monotonic() - t0)}")
    return files


def process_main(source: str, settings_dict: dict, queue, cancel, resummarize: bool = False) -> None:
    """Entry point for the UI's worker process."""
    emit = queue.put
    try:
        run_job(source, Settings.from_dict(settings_dict), emit, cancel, resummarize)
    except Cancelled:
        emit({"type": "cancelled"})
    except Exception as exc:
        emit({"type": "error", "message": str(exc), "traceback": traceback.format_exc()})
