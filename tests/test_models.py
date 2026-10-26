import struct

from localscribe import models
from localscribe.config import Settings, story_minutes
from localscribe.estimates import estimate_minutes, reading_summary
from localscribe.story import story_plan


def _fake_whisper(cache, repo, size):
    snap = cache / f"models--{repo.replace('/', '--')}" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "model.bin").write_bytes(b"\0" * size)


def test_only_downloaded_whisper_models_are_listed(tmp_path):
    _fake_whisper(tmp_path, "mobiuslabsgmbh/faster-whisper-large-v3-turbo", 2000)
    (tmp_path / "models--Systran--faster-whisper-small.en" / "snapshots" / "x").mkdir(parents=True)  # no weights
    installed = models.installed_whisper(tmp_path)
    assert [m.key for m in installed] == ["large-v3-turbo"] and installed[0].size_bytes == 2000
    missing = [m.key for m in models.downloadable_whisper(tmp_path)]
    assert "small.en" in missing and "large-v3-turbo" not in missing


def _gguf(path, name):
    b = name.encode()
    body = b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 12) + b"general.name"
    path.write_bytes(body + struct.pack("<I", 8) + struct.pack("<Q", len(b)) + b)


def test_llms_from_folder_and_selected_file(tmp_path):
    folder, other = tmp_path / "models", tmp_path / "elsewhere"
    folder.mkdir()
    other.mkdir()
    _gguf(folder / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "Meta Llama 3.1 8B Instruct")
    (folder / "gemma-3-4b-it-Q5_K_M.gguf").write_bytes(b"not a real gguf")  # label falls back to the name
    _gguf(other / "custom.gguf", "Custom Model")
    found = models.installed_llms(str(other / "custom.gguf"), folder)
    # Folder models sorted by label (case-insensitive); a file selected elsewhere comes last.
    assert [m.label for m in found] == ["gemma 3 4b it · Q5_K_M", "Meta Llama 3.1 8B Instruct · Q4_K_M",
                                        "Custom Model"]
    assert models.pretty_size(4_920_000_000) == "4.9 GB" and models.pretty_size(484_000_000) == "484 MB"


def test_story_minutes_and_plan():
    assert story_minutes("short") == 5 and story_minutes("long") == 30 and story_minutes(20) == 20
    assert Settings.from_dict({"story_length": "medium"}).story_minutes == 15  # v2 settings file
    assert story_plan(5) == (825, 3) and story_plan(15) == (2475, 4) and story_plan(30) == (4950, 9)


def test_reading_summary_estimates_then_uses_real_files(tmp_path):
    assert estimate_minutes("story", None, 15) == 15
    text = reading_summary(["story", "summary"], 2 * 3600, 15)
    assert text.startswith("About ") and "Engaging Story 15" in text and "Summary 5" in text
    (tmp_path / "story.md").write_text("word " * 330, encoding="utf-8")  # 2 min read aloud
    (tmp_path / "summary.md").write_text("word " * 440, encoding="utf-8")  # 2 min reading
    assert reading_summary(["story", "summary"], 7200, 15, tmp_path) == \
        "Total 4 min of reading · Engaging Story 2 · Summary 2"
    assert reading_summary([], None, 15) == "Transcript only"
