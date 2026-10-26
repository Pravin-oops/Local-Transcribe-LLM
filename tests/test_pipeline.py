"""End-to-end pipeline run with real ffmpeg decoding and fake models."""
import threading
import wave

import numpy as np
import pytest

from localscribe.config import Settings
from localscribe.pipeline import run_job
from localscribe.transcriber import Cancelled

from fakes import FakeBackend, FakeTranscriber, fake_diarizer


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
    settings = _settings(tmp_path, outputs=[])
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


def _model(tmp_path):
    model = tmp_path / "fake-llm.gguf"
    model.write_bytes(b"x")
    return str(model)


def test_lecture_preset_writes_summary_and_notes_and_rerun_skips_work(wav_file, tmp_path):
    settings = _settings(tmp_path, llm_path=_model(tmp_path))
    settings.apply_preset("lecture")
    backend = FakeBackend()
    result = run_job(wav_file, settings, lambda e: None,
                     transcriber_factory=lambda *a: FakeTranscriber([]),
                     backend_factory=lambda *a: backend)
    assert set(result["files"]) >= {"summary.md", "notes.md"} and "story.md" not in result["files"]
    summary = open(result["files"]["summary.md"], encoding="utf-8").read()
    notes = open(result["files"]["notes.md"], encoding="utf-8").read()
    assert "## Summary" in summary and "### Chapters" in summary and "1. Why?" in notes
    assert not {"beats", "outline", "part"} & set(backend.calls)  # no story work for a lecture

    calls = []
    run_job(wav_file, settings, lambda e: None,
            transcriber_factory=lambda *a: FakeTranscriber(calls),
            backend_factory=lambda *a: pytest.fail("outputs should be reused"))
    assert calls == []


def test_movie_preset_detects_speakers_and_writes_only_the_story(wav_file, tmp_path):
    settings = _settings(tmp_path, llm_path=_model(tmp_path))
    settings.apply_preset("movie")
    settings.story_length = "short"
    backend, diarized = FakeBackend(), []
    result = run_job(wav_file, settings, lambda e: None,
                     transcriber_factory=lambda *a: FakeTranscriber([]),
                     backend_factory=lambda *a: backend, diarizer_factory=fake_diarizer(diarized))
    assert list(result["files"])[-1] == "story.md" and "summary.md" not in result["files"]
    assert not {"map", "summary", "notes"} & set(backend.calls)  # no summary work for a movie
    assert backend.calls.count("part") == 3                      # short story = 3 parts
    story = open(result["files"]["story.md"], encoding="utf-8").read()
    assert story.startswith("# The Long Night") and "## Chapter 1 Title" in story
    transcript = open(result["files"]["transcript.txt"], encoding="utf-8").read()
    assert "Speaker 1:" in transcript and "Speaker 2:" in transcript
    assert len(diarized) == 1

    assert backend.part_words == [333, 333, 333]

    # Speakers and beats are cached: a longer story only rewrites the parts. With
    # just 3 scenes in 3 minutes of audio it keeps 3 parts, each longer, capped
    # rather than padded with invented events.
    settings.story_length = "medium"
    longer = FakeBackend()
    run_job(wav_file, settings, lambda e: None, transcriber_factory=lambda *a: FakeTranscriber([]),
            backend_factory=lambda *a: longer, diarizer_factory=fake_diarizer(diarized))
    assert len(diarized) == 1 and "beats" not in longer.calls
    assert longer.part_words == [833, 833, 833]

    # Different speaker labels change the transcript, so the story is redone even
    # though model, tone and length are unchanged. Caches key on content: the
    # transcript is re-read, and since the fake extracts the same events again,
    # the outline and written parts are reused.
    settings.num_speakers = 1  # new diarizer cache key -> diarizer runs again
    relabel = FakeBackend()
    one_voice = lambda samples, *a: [{"start": 0.0, "end": len(samples) / 16000, "speaker": 0}]
    run_job(wav_file, settings, lambda e: None, transcriber_factory=lambda *a: FakeTranscriber([]),
            backend_factory=lambda *a: relabel, diarizer_factory=one_voice)
    assert relabel.calls == ["beats"]


def test_adding_story_to_finished_lecture_only_computes_the_story(wav_file, tmp_path):
    settings = _settings(tmp_path, llm_path=_model(tmp_path))
    settings.apply_preset("lecture")
    run_job(wav_file, settings, lambda e: None, transcriber_factory=lambda *a: FakeTranscriber([]),
            backend_factory=lambda *a: FakeBackend())
    settings.outputs = ["summary", "notes", "story"]
    backend = FakeBackend()
    result = run_job(wav_file, settings, lambda e: None, transcriber_factory=lambda *a: FakeTranscriber([]),
                     backend_factory=lambda *a: backend)
    assert not {"map", "summary", "notes"} & set(backend.calls) and "part" in backend.calls
    assert {"summary.md", "notes.md", "story.md"} <= set(result["files"])


def test_changing_whisper_model_invalidates_parts(wav_file, tmp_path):
    settings = _settings(tmp_path, outputs=[])
    run_job(wav_file, settings, lambda e: None, transcriber_factory=lambda *a: FakeTranscriber([]))
    settings.whisper_model = "small.en"
    calls = []
    run_job(wav_file, settings, lambda e: None, transcriber_factory=lambda *a: FakeTranscriber(calls))
    assert len(calls) == 6
