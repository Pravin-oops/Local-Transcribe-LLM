"""Decode any audio/video file to 16 kHz mono PCM and plan chunk boundaries.

Decoding uses PyAV (installed with faster-whisper), which bundles its own
FFmpeg libraries, so no ffmpeg.exe is needed. The decoded audio is streamed
to raw int16 PCM and memory-mapped, so a 3-hour file (~330 MB) never has to
sit in RAM as float32 (~690 MB). Chunks are cut at the quietest half-second
near each target length, so words are not split.
"""
import os
from pathlib import Path

import av
import numpy as np

SAMPLE_RATE = 16_000

AUDIO_EXTENSIONS = (
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".amr",
    ".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v",
)


class AudioError(RuntimeError):
    pass


def probe_duration(src: Path):
    """Duration in seconds, or None if the container does not say."""
    try:
        with av.open(str(src)) as container:
            return container.duration / av.time_base if container.duration else None
    except av.error.FFmpegError:
        return None


def convert_to_pcm(src: Path, dst: Path, on_progress=None, cancel=None) -> None:
    """Decode ``src`` to raw 16 kHz mono s16le at ``dst`` (written atomically).

    ``on_progress(fraction)`` is called as decoding advances; ``cancel.is_set()``
    aborts it, leaving no output file.
    """
    tmp = dst.with_suffix(".part")
    try:
        container = av.open(str(src))
    except av.error.FFmpegError as exc:
        raise AudioError(f"Could not open {src.name}: {exc}") from exc
    cancelled = False
    with container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise AudioError(f"{src.name} has no audio track")
        duration = container.duration / av.time_base if container.duration else None
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        with open(tmp, "wb") as out:
            for packet in container.demux(stream):
                if cancel is not None and cancel.is_set():
                    cancelled = True
                    break
                try:
                    frames = packet.decode()
                except av.error.InvalidDataError:
                    continue  # skip a corrupt packet rather than fail the whole file
                for frame in frames:
                    for resampled in resampler.resample(frame):
                        out.write(resampled.to_ndarray().tobytes())
                    if on_progress and duration and frame.time is not None:
                        on_progress(min(1.0, frame.time / duration))
            if not cancelled:
                for resampled in resampler.resample(None):  # flush buffered samples
                    out.write(resampled.to_ndarray().tobytes())
    if cancelled:
        tmp.unlink(missing_ok=True)
        return
    os.replace(tmp, dst)


def open_pcm(path: Path) -> np.ndarray:
    """Memory-map the PCM file as int16 samples."""
    if path.stat().st_size < 2:
        raise AudioError("Decoded audio is empty")
    return np.memmap(path, dtype=np.int16, mode="r")


def to_float(samples: np.ndarray) -> np.ndarray:
    return np.asarray(samples, dtype=np.float32) / 32768.0


def _quietest_offset(segment: np.ndarray, prefer: int = None, frame: int = 1600, smooth: int = 5) -> int:
    """Sample offset of a quiet ~0.5 s stretch (5 frames of 0.1 s), nearest ``prefer``.

    Stretches within 10% of the floor-to-median range count as equally quiet,
    so among several pauses the one closest to the preferred point wins.
    """
    n_frames = len(segment) // frame
    if n_frames < smooth:
        return len(segment) // 2
    x = np.asarray(segment[: n_frames * frame], dtype=np.float32).reshape(n_frames, frame)
    # "valid" avoids the zero-padded edges that would make the window borders look silent.
    energy = np.convolve((x * x).mean(axis=1), np.ones(smooth) / smooth, mode="valid")
    floor = energy.min()
    quiet = np.flatnonzero(energy <= floor + 0.1 * (np.median(energy) - floor))
    target = (len(segment) // 2 if prefer is None else prefer) // frame - smooth // 2
    best = int(quiet[np.argmin(np.abs(quiet - target))])
    return (best + smooth // 2) * frame + frame // 2


def plan_chunks(samples: np.ndarray, target_seconds: float, search_seconds: float = 30.0,
                sample_rate: int = SAMPLE_RATE):
    """Split into chunks of about ``target_seconds``, cutting at quiet points.

    Returns a list of ``[start_sample, end_sample]`` pairs that cover every
    sample exactly once. The final chunk may run up to 25% over the target
    rather than leaving a tiny tail.
    """
    n = len(samples)
    target = max(1, int(target_seconds * sample_rate))
    search = int(search_seconds * sample_rate)
    chunks, start = [], 0
    while n - start > target + target // 4:
        ideal = start + target
        lo = max(start + target // 2, ideal - search)
        hi = min(n, ideal + search)
        cut = lo + _quietest_offset(samples[lo:hi], prefer=ideal - lo)
        chunks.append([start, cut])
        start = cut
    chunks.append([start, n])
    return chunks
