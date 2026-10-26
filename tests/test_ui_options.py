"""Options panel and main window behaviour, headless."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from localscribe import models
from localscribe.config import Settings
from localscribe.ui.options_panel import OptionsPanel

QtWidgets = pytest.importorskip("PySide6.QtWidgets")


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def fake_models(monkeypatch, tmp_path):
    """Pretend exactly one Whisper model and two GGUF files are installed."""
    whisper = [models.ModelInfo("large-v3-turbo", "large-v3-turbo", 1_620_000_000, "best balance")]
    monkeypatch.setattr(models, "installed_whisper", lambda cache=None: list(whisper))
    monkeypatch.setattr(models, "downloadable_whisper", lambda cache=None: [
        models.ModelInfo("small.en", "small.en", 484_000_000, "fast, English")])
    llms = [models.ModelInfo(str(tmp_path / "a.gguf"), "Gemma 3 4B · Q5_K_M", 2_500_000_000),
            models.ModelInfo(str(tmp_path / "b.gguf"), "Meta Llama 3.1 8B Instruct · Q4_K_M", 4_900_000_000)]
    monkeypatch.setattr(models, "installed_llms", lambda selected="", folder=None: list(llms))
    return llms


def _state(panel):
    s = Settings()
    panel.apply_to(s)
    return s


def test_model_pickers_list_only_installed_models(app, fake_models):
    panel = OptionsPanel(Settings(llm_path=fake_models[1].key))
    assert panel.whisper.name.text() == "large-v3-turbo" and panel.whisper.size.text() == "1.6 GB"
    assert panel.llm.name.text() == "Meta Llama 3.1 8B Instruct · Q4_K_M"
    menu = panel.llm.build_menu()
    texts = [a.text() for a in menu.actions() if not a.isSeparator()]
    assert texts[:2] == ["Gemma 3 4B · Q5_K_M\t2.5 GB", "Meta Llama 3.1 8B Instruct · Q4_K_M\t4.9 GB"]
    assert [a.isChecked() for a in menu.actions()[:2]] == [False, True]
    assert "Add a model file…" in texts
    whisper_menu = panel.whisper.build_menu()
    download = next(a for a in whisper_menu.actions() if a.text() == "Download another model").menu()
    assert [a.text() for a in download.actions()] == ["small.en  ·  fast, English\t484 MB"]

    panel.llm._choose(fake_models[0].key)  # picking from the menu updates the settings
    assert _state(panel).llm_path == fake_models[0].key


def test_preset_ticks_its_outputs_and_chips_switch_preset(app):
    panel = OptionsPanel(Settings())  # lecture: summary + notes
    panel.preset.buttons["movie"].click()
    s = _state(panel)
    assert (s.content_type, s.outputs, s.diarize) == ("movie", ["story"], True)
    assert not panel.story_row.isHidden()
    panel.chips["story"].setChecked(False)
    panel.chips["summary"].setChecked(True)
    assert _state(panel).content_type == "podcast"      # summary + speakers = podcast preset
    panel.chips["notes"].setChecked(True)
    s = _state(panel)
    assert s.content_type == "custom" and s.outputs == ["summary", "notes"]
    assert panel.story_row.isHidden()


def test_story_minutes_round_trip(app):
    s = Settings(outputs=["story"], content_type="movie", diarize=True, story_tone="bedtime", story_minutes=20)
    panel = OptionsPanel(s)
    assert (_state(panel).story_tone, _state(panel).story_minutes) == ("bedtime", 20)
    panel = OptionsPanel(Settings(outputs=["story"], story_minutes=12))  # custom value from the CLI
    assert _state(panel).story_minutes == 12 and panel.minutes.currentText() == "12 min read"


def test_opening_subtitles_selects_movie_preset_and_shows_reading_time(app, tmp_path, monkeypatch):
    from localscribe.ui import main_window
    monkeypatch.setattr(main_window, "load_settings", lambda: Settings(output_dir=str(tmp_path / "out")))
    monkeypatch.setattr(main_window, "save_settings", lambda s: None)  # never touch the real settings file
    srt = tmp_path / "film.srt"
    srt.write_text("1\n00:00:01,000 --> 01:30:00,000\nTOMAS: Hold the line.\n", encoding="utf-8")
    win = main_window.MainWindow()
    assert win.settings.content_type == "lecture"
    win._open_source(srt)
    assert (win.settings.content_type, win.settings.outputs) == ("movie", ["story"])
    assert not win.panel.speakers.isEnabled()
    assert win.drop.title.text() == "film.srt" and win.drop.detail.text().startswith("Subtitles · 1 line · ")
    assert win.panel.reading.text() == "About 15 min of reading · Engaging Story 15"
    win.panel.chips["summary"].setChecked(True)   # a refresh keeps the user's choice
    assert "summary" in win.settings.outputs and "Summary" in win.panel.reading.text()
    win.close()
