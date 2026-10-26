"""Output picker behaviour, headless."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from localscribe.config import Settings
from localscribe.ui.output_picker import OutputPicker

QtWidgets = pytest.importorskip("PySide6.QtWidgets")


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _state(picker):
    s = Settings()
    picker.apply_to(s)
    return s


def test_preset_ticks_its_outputs(app):
    picker = OutputPicker(Settings())
    picker.preset.setCurrentIndex(picker.preset.findData("movie"))
    s = _state(picker)
    assert (s.content_type, s.outputs, s.diarize) == ("movie", ["story"], True)
    assert not picker.story_row.isHidden()


def test_manual_ticks_switch_to_matching_preset_or_custom(app):
    picker = OutputPicker(Settings())  # lecture: summary + notes
    picker.boxes["notes"].setChecked(False)
    picker.speakers.setChecked(True)
    assert _state(picker).content_type == "podcast"      # summary + speakers = podcast preset
    picker.boxes["story"].setChecked(True)
    s = _state(picker)
    assert s.content_type == "custom" and s.outputs == ["summary", "story"]


def test_story_options_round_trip(app):
    s = Settings(outputs=["story"], content_type="movie", diarize=True, story_tone="bedtime", story_length="long")
    picker = OutputPicker(s)
    assert (_state(picker).story_tone, _state(picker).story_length) == ("bedtime", "long")
    assert picker.story_row.isVisibleTo(picker)
