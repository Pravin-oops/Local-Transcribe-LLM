"""Story translation: Markdown handling and the pipeline's last stage, with a fake translator."""
import threading

import pytest

from localscribe import translator
from localscribe.config import Settings
from localscribe.pipeline import run_job
from localscribe.transcriber import Cancelled

from fakes import FakeBackend

STORY = """# The Long Night

*A captain must choose. Nobody is coming.*

*Genre: thriller · Narration: matched to the film · 120 words (about 1 min read aloud) · Retold from film.srt*

**Cast**

- **Mara** — The captain. She never sleeps.

---

## Chapter 1: The Storm

The night was long. "Hold the line!" Mara said. Nobody moved…

Then the lamp went out.
"""


class FakeTranslator:
    name, device = "fake-nllb", "cpu"

    def __init__(self, calls: list, cancel_after: int = None, cancel=None):
        self.calls, self.cancel_after, self.cancel = calls, cancel_after, cancel
        self.closed = False

    def translate(self, texts, language, on_progress=None, cancel=None):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        self.calls.append(list(texts))
        if self.cancel_after and len(self.calls) >= self.cancel_after:
            self.cancel.set()
        return [f"{language}<{t}>" for t in texts]

    def close(self):
        self.closed = True


def test_sentences_split_after_closing_quotes_and_ellipses():
    assert translator.split_sentences('The night was long. "Hold the line!" Mara said. Nobody moved… Then dark.') == [
        "The night was long.", '"Hold the line!"', "Mara said.", "Nobody moved…", "Then dark."]
    # A lower-case continuation stays with its quote, so the model sees who is speaking.
    assert translator.split_sentences('"Is anyone here?" she called. Nobody answered.') == [
        '"Is anyone here?" she called.', "Nobody answered."]
    # Repairs to the model's output: a dropped quotation mark, a colon written as "ஃ".
    assert translator._tidy('"Is anyone here?"', '"யாரும் இங்கே இருக்கிறார்களா?') == \
        '"யாரும் இங்கே இருக்கிறார்களா?"'
    assert translator._tidy("Nobody answered.", "யாரும் பதில் சொல்லவில்லை.") == "யாரும் பதில் சொல்லவில்லை."
    assert translator._tidy('She asked: "Why?"', 'அவள் கேட்டாள்ஃ "ஏன்?"') == 'அவள் கேட்டாள்: "ஏன்?"'
    assert translator._tidy("That is all.", "அஃது எல்லாம்.") == "அஃது எல்லாம்."  # a real aytham stays


def test_markdown_structure_names_and_details_line_survive_translation():
    asked = []

    def fake(texts):
        asked.extend(texts)
        return [f"<{t}>" for t in texts]

    out = translator.translate_markdown(STORY, fake, {"Cast": "கதாபாத்திரங்கள்"}, note="Tamil translation by X")
    assert out.split("\n") == [
        "# <The Long Night>", "",
        "*<A captain must choose.> <Nobody is coming.>*", "",
        "*Genre: thriller · Narration: matched to the film · 120 words (about 1 min read aloud) · "
        "Retold from film.srt · Tamil translation by X*", "",
        "**கதாபாத்திரங்கள்**", "",
        "- **Mara** — <The captain.> <She never sleeps.>", "",   # the name stays as written
        "---", "",
        "## <Chapter 1: The Storm>", "",
        '<The night was long.> <"Hold the line!"> <Mara said.> <Nobody moved…>', "",
        "<Then the lamp went out.>", ""]
    assert "Cast" not in asked and not any("Genre" in t for t in asked)


def _settings(tmp_path, **kw):
    model = tmp_path / "fake-llm.gguf"
    model.write_bytes(b"x")
    settings = Settings(output_dir=str(tmp_path / "out"), llm_path=str(model), story_minutes=5, **kw)
    settings.apply_preset("movie")
    return settings


@pytest.fixture
def srt(tmp_path):
    path = tmp_path / "film.srt"
    cues = ["ELENA: Hello? Is anyone up here?", "TOMAS: The radio is right.", "Then the lamp has to work.",
            "The ship is turning!", "They see us.", "We made it home."]
    path.write_text("\n\n".join(f"{i + 1}\n00:00:{i * 9:02d},000 --> 00:00:{i * 9 + 8:02d},000\n{c}"
                                for i, c in enumerate(cues)), encoding="utf-8")
    return path


def test_story_is_translated_last_and_reused_until_the_story_changes(srt, tmp_path):
    settings = _settings(tmp_path, translate_to="ta")
    calls, events, made = [], [], []

    def factory(*a):
        made.append(FakeTranslator(calls))
        return made[-1]

    result = run_job(srt, settings, events.append, backend_factory=lambda *a: FakeBackend(),
                     translator_factory=factory)
    assert list(result["files"])[-2:] == ["story.md", "story.ta.md"]
    english = open(result["files"]["story.md"], encoding="utf-8").read()
    tamil = open(result["files"]["story.ta.md"], encoding="utf-8").read()
    assert tamil.startswith("# ta<The Long Night>") and "## ta<Chapter 1 Title>" in tamil
    assert "ta<The night was long.>" in tamil and "**கதாபாத்திரங்கள்**" in tamil
    assert "Tamil translation by NLLB-200 1.3B*" in tamil
    assert len(tamil.split("\n")) == len(english.split("\n"))  # the two files line up
    assert made[0].closed
    stages = [e["stage"] for e in events if e["type"] == "progress"]
    assert stages[-1] == "translate" and stages.index("translate") > stages.index("story")
    assert [e for e in events if e["type"] == "progress"][-1]["fraction"] == pytest.approx(1.0)

    # Same story: nothing is loaded or translated again.
    run_job(srt, settings, lambda e: None, backend_factory=lambda *a: pytest.fail("story is reused"),
            translator_factory=lambda *a: pytest.fail("translation is reused"))

    # A different tone rewrites the story; only sentences not seen before are translated.
    settings.story_tone = "bedtime"
    before = sum(len(c) for c in calls)
    run_job(srt, settings, lambda e: None, backend_factory=lambda *a: FakeBackend(), translator_factory=factory)
    assert sum(len(c) for c in calls) == before  # the fake LLM wrote the same sentences

    # Without the option, or without the story output, nothing is translated.
    settings.translate_to = ""
    plain = run_job(srt, settings, lambda e: None, translator_factory=lambda *a: pytest.fail("not selected"))
    assert "story.ta.md" not in plain["files"]
    settings.translate_to, settings.outputs = "ta", ["summary"]
    run_job(srt, settings, lambda e: None, backend_factory=lambda *a: FakeBackend(),
            translator_factory=lambda *a: pytest.fail("only the story is translated"))


def test_cancelled_translation_resumes_with_the_sentences_already_done(srt, tmp_path, monkeypatch):
    monkeypatch.setattr(translator, "BATCH", 3)
    settings = _settings(tmp_path, translate_to="ta")
    cancel, first = threading.Event(), []
    with pytest.raises(Cancelled):
        run_job(srt, settings, lambda e: None, cancel, backend_factory=lambda *a: FakeBackend(),
                translator_factory=lambda *a: FakeTranslator(first, cancel_after=2, cancel=cancel))
    assert len(first) == 2

    second = []
    result = run_job(srt, settings, lambda e: None, backend_factory=lambda *a: pytest.fail("story is reused"),
                     translator_factory=lambda *a: FakeTranslator(second))
    done_before = {t for batch in first for t in batch}
    assert second and not done_before & {t for batch in second for t in batch}
    assert "ta<The night was long.>" in open(result["files"]["story.ta.md"], encoding="utf-8").read()
