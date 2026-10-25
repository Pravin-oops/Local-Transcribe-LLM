"""End-to-end pipeline run with real ffmpeg decoding and fake models."""
import threading
import wave

import numpy as np
import pytest

from localscribe.config import Settings
from localscribe.pipeline import run_job
from localscribe.transcriber import Cancelled

from fakes import FakeBackend, FakeTranscriber


@pytest.fixture
def wav_file(tmp_path):
    """3 minutes of tone bursts separated by short silences."""
    sr, seconds = 16000, 180
    t = np.arange(sr * seconds) / sr
    signal = (np.sin(2 * np.pi * 220 * t) * 9000).astype(np.int16)
    for start in range(25, seconds, 30):
        signal[start * sr: (start + 1) * sr] = 0
    path = tmp_path / "lecture.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(signal.tobytes())
    return path


def _settings(tmp_path, **kw):
    return Settings(output_dir=str(tmp_path / "out"), chunk_minutes=0.5, **kw)


def test_cancel_then_resume_transcribes_only_missing_parts(wav_file, tmp_path):
    settings = _settings(tmp_path, summarize=False)
    cancel = threading.Event()
    first_calls = []
    with pytest.raises(Cancelled):
        run_job(wav_file, settings, lambda e: None, cancel,
                transcriber_factory=lambda *a: FakeTranscriber(first_calls, cancel_after=2, cancel=cancel))
    assert len(first_calls) == 2

    second_calls, events = [], []
    result = run_job(wav_file, settings, events.append,
                     transcriber_factory=lambda *a: FakeTranscriber(second_calls))
    manifest_chunks = 6  # 180 s / 30 s
    assert len(second_calls) == manifest_chunks - 2
    assert min(second_calls) > max(first_calls)  # picked up where it stopped

    txt = open(result["files"]["transcript.txt"], encoding="utf-8").read()
    assert txt.startswith("[00:00:00]")
    srt = open(result["files"]["transcript.srt"], encoding="utf-8").read()
    assert srt.count("-->") == manifest_chunks
    fractions = [e["fraction"] for e in events if e["type"] == "progress"]
    assert fractions[-1] == pytest.approx(1.0)


def test_full_run_with_summary_and_rerun_skips_work(wav_file, tmp_path):
    model = tmp_path / "fake-llm.gguf"
    model.write_bytes(b"x")
    settings = _settings(tmp_path, llm_path=str(model))
    backend = FakeBackend()
    result = run_job(wav_file, settings, lambda e: None,
                     transcriber_factory=lambda *a: FakeTranscriber([]),
                     backend_factory=lambda *a: backend)
    summary = open(result["files"]["summary.md"], encoding="utf-8").read()
    assert "## Chapters" in summary and "## Study Notes" in summary and "1. Why?" in summary

    calls = []
    run_job(wav_file, settings, lambda e: None,
            transcriber_factory=lambda *a: FakeTranscriber(calls),
            backend_factory=lambda *a: pytest.fail("summary should be reused"))
    assert calls == []


def test_changing_whisper_model_invalidates_parts(wav_file, tmp_path):
    settings = _settings(tmp_path, summarize=False)
    run_job(wav_file, settings, lambda e: None, transcriber_factory=lambda *a: FakeTranscriber([]))
    settings.whisper_model = "small.en"
    calls = []
    run_job(wav_file, settings, lambda e: None, transcriber_factory=lambda *a: FakeTranscriber(calls))
    assert len(calls) == 6
