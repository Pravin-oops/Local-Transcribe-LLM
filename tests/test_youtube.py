"""YouTube links: caption choice and parsing, the audio fallback and the pipeline, all offline."""
import io
import json
from pathlib import Path

import pytest

from localscribe import youtube
from localscribe.config import Settings
from localscribe.jobs import read_json
from localscribe.pipeline import run_job

from fakes import FakeBackend, FakeTranscriber

VID = "aircAruvnKk"
URL = f"https://www.youtube.com/watch?v={VID}"

# Auto-generated captions: word-timed lines that overlap, newline-only events, sound tags.
AUTO = {"events": [
    {"tStartMs": 0, "dDurationMs": 60000, "id": 1},
    {"tStartMs": 4280, "dDurationMs": 4080, "segs": [{"utf8": "this"}, {"utf8": " is f(x)", "tOffsetMs": 480}]},
    {"tStartMs": 6830, "dDurationMs": 1530, "aAppend": 1, "segs": [{"utf8": "\n"}]},
    {"tStartMs": 6840, "dDurationMs": 4400, "segs": [{"utf8": ">> and"}, {"utf8": " rendered\nsmall"}]},
    {"tStartMs": 12000, "dDurationMs": 2000, "segs": [{"utf8": "[Music]"}]},
]}


def test_recognises_video_links_only():
    for link in (URL, "youtube.com/watch?list=PL1&v=" + VID, f"https://youtu.be/{VID}?t=42",
                 f"https://m.youtube.com/shorts/{VID}", f"https://www.youtube.com/live/{VID}?feature=share",
                 f"  https://music.youtube.com/watch?v={VID}&si=abc  "):
        assert youtube.video_id(link) == VID, link
    for other in ("lecture.mp3", r"C:\videos\film.srt", Path("notes") / "watch?v=aircAruvnKk",
                  "https://www.youtube.com/playlist?list=PL1", "https://vimeo.com/123456789", ""):
        assert not youtube.is_url(other), other


def test_creator_captions_come_first_then_the_spoken_language_auto_track():
    def track(name):
        return [{"ext": "vtt", "url": f"{name}.vtt"}, {"ext": "json3", "url": f"{name}.json3"}]
    info = {"subtitles": {"fr": track("m-fr"), "en-GB": track("m-en-GB")},
            "automatic_captions": {"en": track("a-en"), "en-orig": track("a-en-orig"), "de": track("a-de")}}
    assert youtube.caption_tracks(info, "en") == [("manual", "en-GB", "m-en-GB.json3"),
                                                  ("auto", "en-orig", "a-en-orig.json3"),
                                                  ("auto", "en", "a-en.json3")]
    assert youtube.caption_tracks({"subtitles": {"fr": track("m-fr")}}, "en") == []


def test_auto_captions_become_clean_cues_and_survive_the_saved_file(tmp_path):
    cues = youtube.parse_json3(AUTO)
    # Scrolling lines are cut where the next begins; ">>" and "[Music]" are dropped.
    assert cues == [{"start": 4.28, "end": 6.84, "text": "this is f(x)"},
                    {"start": 6.84, "end": 11.24, "text": "and rendered small"}]
    path = tmp_path / f"Talk [{VID}].vtt"
    path.write_text(youtube.to_vtt(cues, URL, "auto"), encoding="utf-8")
    assert youtube.caption_kind(path) == "auto-generated"
    assert youtube.load_captions(path) == cues  # "f(x)" is text, not a movie sound cue


class FakeYdl:
    """Stands in for yt_dlp.YoutubeDL: serves caption tracks and 'downloads' audio."""

    def __init__(self, info, tracks, options):
        self.info, self.tracks, self.options = info, tracks, options

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=False):
        if not download:
            return self.info
        path = Path(self.options["outtmpl"].replace("%(ext)s", "webm"))
        path.write_bytes(b"opus")
        for hook in self.options["progress_hooks"]:
            hook({"status": "downloading", "downloaded_bytes": 2**20, "total_bytes": 2**21})
        return {**self.info, "requested_downloads": [{"filepath": str(path)}]}

    def urlopen(self, url):
        return io.BytesIO(self.tracks[url])


def _online(monkeypatch, info, tracks=None):
    monkeypatch.setattr(youtube, "_session", lambda **options: FakeYdl(info, tracks or {}, options))


def test_fetch_saves_captions_and_reuses_them_offline(tmp_path, monkeypatch):
    info = {"id": VID, "title": 'What is a "neural network"? | Chapter 1',
            "automatic_captions": {"en": [{"ext": "json3", "url": "auto"}]}}
    _online(monkeypatch, info, {"auto": json.dumps(AUTO).encode()})
    said = []
    got = youtube.fetch(f"https://youtu.be/{VID}", tmp_path / "yt", "en", said.append)
    assert got.kind == "captions" and got.path.name == f"What is a neural network Chapter 1 [{VID}].vtt"
    assert "auto-generated captions found" in got.notes[-1] and "Checking YouTube for captions" in said

    monkeypatch.setattr(youtube, "_session", lambda **o: pytest.fail("saved captions need no network"))
    again = youtube.fetch(URL, tmp_path / "yt")
    assert again.path == got.path and again.kind == "captions"
    assert youtube.cached(VID, tmp_path / "yt") == got.path


def test_fetch_downloads_audio_when_there_are_no_usable_captions(tmp_path, monkeypatch):
    info = {"id": VID, "title": "Silent film", "subtitles": {"en": [{"ext": "json3", "url": "empty"}]}}
    _online(monkeypatch, info, {"empty": b'{"events": []}'})
    said = []
    got = youtube.fetch(URL, tmp_path / "yt", "en", said.append)
    assert got.kind == "audio" and got.path.name == f"Silent film [{VID}].webm" and got.path.is_file()
    assert "downloading the audio" in got.notes[-1]
    assert "Downloading audio 50% (1.0 of 2.0 MB)" in said


def _settings(tmp_path, **kw):
    model = tmp_path / "fake-llm.gguf"
    model.write_bytes(b"x")
    return Settings(output_dir=str(tmp_path / "out"), chunk_minutes=0.5, llm_path=str(model), **kw)


def test_link_with_captions_is_summarized_without_whisper(tmp_path, monkeypatch):
    lines = [{"tStartMs": i * 5000, "dDurationMs": 5000, "segs": [{"utf8": f"Point {i} about f(x)."}]}
             for i in range(12)]
    info = {"id": VID, "title": "Neural networks", "subtitles": {"en": [{"ext": "json3", "url": "manual"}]}}
    _online(monkeypatch, info, {"manual": json.dumps({"events": lines}).encode()})
    settings = _settings(tmp_path, outputs=["summary"], diarize=True)
    backend, events = FakeBackend(), []
    result = run_job(URL, settings, events.append,
                     transcriber_factory=lambda *a: pytest.fail("no Whisper when captions exist"),
                     diarizer_factory=lambda *a: pytest.fail("no speaker detection for captions"),
                     backend_factory=lambda *a: backend)
    summary = open(result["files"]["summary.md"], encoding="utf-8").read()
    assert summary.startswith(f"# Neural networks [{VID}]") and "From creator-provided YouTube captions" in summary
    assert "Point 3 about f(x)." in open(result["files"]["transcript.txt"], encoding="utf-8").read()
    assert read_json(Path(result["job_dir"]) / "manifest.json")["source_url"] == URL
    assert events[0] == {"type": "progress", "stage": "fetch", "fraction": 0.0,
                         "message": "Checking YouTube for captions", "eta": None}
    assert any("creator-provided captions found" in e["message"] for e in events if e["type"] == "log")

    # The same link again is the same job: nothing is fetched or summarized twice.
    monkeypatch.setattr(youtube, "_session", lambda **o: pytest.fail("saved captions need no network"))
    again = run_job(URL, settings, lambda e: None, backend_factory=lambda *a: pytest.fail("summary is reused"))
    assert again["job_dir"] == result["job_dir"]


def test_link_without_captions_is_transcribed_from_the_downloaded_audio(tmp_path, monkeypatch, wav_bytes):
    class AudioYdl(FakeYdl):
        def extract_info(self, url, download=False):
            if download:
                path = Path(self.options["outtmpl"].replace("%(ext)s", "wav"))
                path.write_bytes(wav_bytes)
                return {**self.info, "requested_downloads": [{"filepath": str(path)}]}
            return self.info
    monkeypatch.setattr(youtube, "_session", lambda **o: AudioYdl({"id": VID, "title": "No captions"}, {}, o))
    calls, events = [], []
    result = run_job(URL, _settings(tmp_path, outputs=["summary"]), events.append,
                     transcriber_factory=lambda *a: FakeTranscriber(calls),
                     backend_factory=lambda *a: FakeBackend())
    assert len(calls) == 2  # 60 s of audio in 30 s parts went through Whisper
    assert "Transcribed with" in open(result["files"]["summary.md"], encoding="utf-8").read()
    assert any("this video has no captions" in e["message"] for e in events if e["type"] == "log")


@pytest.fixture
def wav_bytes(tmp_path):
    import wave

    import numpy as np
    sr, seconds = 16000, 60
    signal = (np.sin(2 * np.pi * 220 * np.arange(sr * seconds) / sr) * 9000).astype(np.int16)
    signal[29 * sr: 30 * sr] = 0
    path = tmp_path / "tone.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(signal.tobytes())
    return path.read_bytes()
