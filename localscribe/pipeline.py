"""Run one file through prepare -> transcribe -> [speakers] -> selected outputs.

Only the outputs the user selected are computed (summary, study notes,
Engaging Story), and every stage resumes from what is already on disk.
Everything reports through ``emit(event: dict)`` so the same code drives the
CLI and the desktop UI (where events cross a multiprocessing queue). Events:

    {"type": "job", "job_dir"}
    {"type": "progress", "stage", "fraction", "message", "eta"}
    {"type": "segments", "items": [{"start", "end", "text"}, ...]}
    {"type": "log", "level", "message"}
    {"type": "done", "job_dir", "files"}
    {"type": "cancelled"} / {"type": "error", "message", "traceback"}
"""
import hashlib
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

from localscribe import audio, hardware, prompts, subtitles
from localscribe.subtitles import is_subtitle
from localscribe.config import OUTPUTS, Settings
from localscribe.jobs import Job, read_json, write_json_atomic
from localscribe.merger import write_transcripts
from localscribe.story import render_story, write_story
from localscribe.summarizer import LlamaBackend, auto_gpu_layers, render_notes, render_summary, summarize
from localscribe.timefmt import hms
from localscribe.transcriber import Cancelled, Transcriber

SR = audio.SAMPLE_RATE

# Relative cost of each stage, used to map stage progress onto one bar.
STAGE_WEIGHTS = {"prepare": 1, "transcribe": 12, "diarize": 3, "summarize": 6, "story": 8}


class _NeverCancel:
    @staticmethod
    def is_set() -> bool:
        return False


class _Reporter:
    """Maps per-stage progress onto one 0..1 bar and mirrors logs to the job folder."""

    def __init__(self, emit, stages: list):
        self.emit = emit
        self.log_path = None
        total = sum(STAGE_WEIGHTS[s] for s in stages)
        self.spans, start = {}, 0.0
        for s in stages:
            end = start + STAGE_WEIGHTS[s] / total
            self.spans[s] = (start, end)
            start = end
        self.last = 0.0

    def progress(self, stage: str, fraction: float, message: str, eta: float = None) -> None:
        lo, hi = self.spans.get(stage, (self.last, self.last))
        self.last = lo + (hi - lo) * max(0.0, min(1.0, fraction))
        self.emit({"type": "progress", "stage": stage, "fraction": self.last,
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


def _default_diarizer(samples, settings: Settings, on_progress, cancel, on_status):
    from localscribe import diarizer
    return diarizer.diarize(samples, settings.num_speakers, hardware.detect().physical_cores,
                            on_progress, cancel, on_status=on_status)


def run_job(source, settings: Settings, emit, cancel=None, resummarize: bool = False,
            transcriber_factory=None, backend_factory=None, diarizer_factory=None) -> dict:
    cancel = cancel or _NeverCancel()
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"Audio file not found: {source}")
    outputs = [o for o in OUTPUTS if o in settings.outputs]
    use_llm = bool(outputs and settings.llm_path)
    from_subtitles = is_subtitle(source)
    if from_subtitles:  # the subtitles are the transcript: no audio, Whisper or diarizer
        stages = ["prepare"]
    else:
        stages = ["prepare", "transcribe"] + (["diarize"] if settings.diarize else [])
    if use_llm:
        stages += (["summarize"] if {"summary", "notes"} & set(outputs) else []) + \
                  (["story"] if "story" in outputs else [])
    report = _Reporter(emit, stages)
    job = Job.for_source(source, Path(settings.output_dir))
    report.log_path = job.output("log.txt")
    emit({"type": "job", "job_dir": str(job.root)})
    report.log(f"Job started for {source} (outputs: {', '.join(outputs) or 'transcript only'})")

    if from_subtitles:
        segments = _read_subtitles(job, source, settings, report, emit)
    else:
        _prepare(job, source, settings, report, cancel)
        segments = _transcribe(job, settings, report, emit, cancel,
                               transcriber_factory or _default_transcriber)
        if settings.diarize:
            segments = _diarize(job, segments, settings, report, cancel,
                                diarizer_factory or _default_diarizer)
    files = write_transcripts(segments, job.root)
    job.set_stage("transcribed", True)
    report.log(f"Transcript written: {len(segments)} segments")

    if outputs and not settings.llm_path:
        report.warn(f"{', '.join(OUTPUTS[o] for o in outputs)} skipped: no LLM model selected in Settings.")
    elif use_llm:
        files.update(_llm_outputs(job, source, segments, settings, outputs, report, cancel, resummarize,
                                  backend_factory or _default_backend))

    result = {"type": "done", "job_dir": str(job.root), "files": files}
    report.log("Job finished")
    emit(result)
    return result


def _read_subtitles(job: Job, source: Path, settings: Settings, report: _Reporter, emit) -> list:
    report.progress("prepare", 0.0, f"Reading subtitles {source.name}")
    segments = subtitles.load(source)
    job.update_manifest(source_type="subtitles", duration=segments[-1]["end"])
    spoken = sum(1 for s in segments if not s.get("sound"))
    report.log(f"Subtitles: {spoken} spoken lines, {len(segments) - spoken} sound cues, "
               f"{hms(segments[-1]['end'])}; transcription skipped")
    if settings.diarize:
        report.log("Speaker detection needs audio; using speaker names written in the subtitles instead")
    emit({"type": "segments", "items": segments})
    report.progress("prepare", 1.0, "Subtitles loaded")
    return segments


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
        _invalidate_outputs(job)
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
        _invalidate_outputs(job)
        job.set_stage("transcribed", False)
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


def _invalidate_outputs(job: Job) -> None:
    """The transcript changed: every LLM result built on it is stale."""
    job.clear_dir(job.notes_dir)
    job.clear_dir(job.output("story"))
    job.update_manifest(outputs_done={})


def _diarize(job: Job, segments: list, settings: Settings, report: _Reporter, cancel, factory) -> list:
    from localscribe.diarizer import ENGINE, assign_speakers

    path = job.output("speakers.json")
    key = {"engine": ENGINE, "num_speakers": settings.num_speakers}
    cached = read_json(path)
    if cached and cached.get("key") == key:
        turns = cached["turns"]
        report.progress("diarize", 1.0, "Speakers already detected")
    else:
        report.progress("diarize", 0.0, "Detecting speakers")
        t0 = time.monotonic()
        samples = audio.to_float(audio.open_pcm(job.pcm_path))
        turns = factory(samples, settings, lambda f: report.progress("diarize", f, "Detecting speakers"),
                        cancel, report.log)
        del samples
        write_json_atomic(path, {"key": key, "turns": turns})
        report.log(f"Detected {len({t['speaker'] for t in turns})} speakers in "
                   f"{hms(time.monotonic() - t0)}")
    return assign_speakers(segments, turns)


def _llm_outputs(job: Job, source: Path, segments: list, settings: Settings, outputs: list,
                 report: _Reporter, cancel, resummarize: bool, factory) -> dict:
    """Produce the selected outputs, skipping any already done with the same settings."""
    llm_path = Path(settings.llm_path)
    if not llm_path.is_file():
        report.warn(f"Outputs skipped: model file not found: {llm_path}")
        return {}
    # Fingerprint of what the outputs are built from: a change of speaker labels,
    # transcript text or prompts must redo them even if the settings look the same.
    basis = json.dumps([[s["start"], s.get("speaker"), s["text"]] for s in segments])
    base = {"model": llm_path.stem, "prompts": prompts.VERSION,
            "transcript": hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]}
    configs = {"summary": base, "notes": base,
               "story": {**base, "tone": settings.story_tone, "minutes": settings.story_minutes}}
    if resummarize:
        _invalidate_outputs(job)
    done = job.load_manifest().get("outputs_done", {})
    files = {f"{o}.md": str(job.output(f"{o}.md")) for o in outputs}
    pending = [o for o in outputs if done.get(o) != configs[o] or not job.output(f"{o}.md").exists()]
    for o in outputs:
        if o not in pending:
            report.log(f"{OUTPUTS[o]} already complete")
    text_outputs = [o for o in pending if o in ("summary", "notes")]
    if not text_outputs and {"summary", "notes"} & set(outputs):
        report.progress("summarize", 1.0, "Summary and notes already complete")
    if not pending:
        if "story" in outputs:
            report.progress("story", 1.0, "Story already complete")
        return files

    report.progress("summarize" if text_outputs else "story", 0.0, f"Loading LLM {llm_path.stem}")
    backend = factory(settings, report.log)
    report.log(f"LLM {backend.name}: context {backend.n_ctx}, GPU layers {getattr(backend, 'gpu_layers', '?')}")
    manifest = job.load_manifest()
    duration = manifest.get("duration", 0)
    basis = ("From subtitles" if manifest.get("source_type") == "subtitles"
             else f"Transcribed with {settings.whisper_model}")
    # *...* rather than _..._: model names like Q4_K_M contain underscores.
    header = f"# {source.stem}\n\n*Duration {hms(duration)} · {basis} · Written by {backend.name}*\n\n"

    def finish(output: str, text: str, t0: float) -> None:
        job.output(f"{output}.md").write_text(text, encoding="utf-8")
        done[output] = configs[output]
        job.update_manifest(outputs_done=done)
        report.log(f"{OUTPUTS[output]} written in {hms(time.monotonic() - t0)}")

    try:
        if text_outputs:
            t0 = time.monotonic()
            result = summarize(segments, backend, job, source.stem, text_outputs,
                               lambda f, m: report.progress("summarize", f, m), cancel, report.warn)
            if "summary" in text_outputs:
                finish("summary", header + render_summary(result["summary"], result["chapters"]), t0)
            if "notes" in text_outputs:
                finish("notes", header + render_notes(result["notes"]), t0)
        if "story" in pending:
            t0 = time.monotonic()
            story = write_story(segments, backend, job, source.stem, settings.story_tone,
                                settings.story_minutes, lambda f, m: report.progress("story", f, m),
                                cancel, report.warn)
            source_line = f"Retold from {source.name} ({hms(duration)}) by {backend.name}"
            finish("story", render_story(story, source_line), t0)
    finally:
        backend.close()
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
