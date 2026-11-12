"""The main page's options card: models, content preset, outputs, story length
and the total reading time, laid out like the sketch:

    Transcribe model   ( large-v3-turbo           1.6 GB  v )
    LLM model          ( Meta Llama 3.1 8B · Q4   4.9 GB  v )
    Content            [ Lecture | Podcast | Movie | Custom ]
    Output             (Summary) (Study notes) (Engaging Story)   (Detect speakers)
    Story              [ Match the film's genre v ]  [ 15 min read v ]   (Translate to Tamil)
    Reading time       About 22 min of reading · Engaging Story 15 · Summary 3 · ...

Picking a preset ticks its outputs; ticking chips by hand switches the preset
to the matching one, or to Custom. Only models found on disk are listed.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox, QGridLayout, QHBoxLayout, QLabel, QWidget

from localscribe import models, translator
from localscribe.config import OUTPUTS, PRESETS, STORY_MINUTES, STORY_TONES, Settings, models_dir
from localscribe.ui import theme
from localscribe.ui.widgets import Card, Chip, ModelPicker, SegmentedControl, icon_label

OUTPUT_TIPS = {
    "summary": "Overview, key points and timestamped chapters",
    "notes": "Key concepts, takeaways, examples and review questions",
    "story": "A captivating narrator script that retells the whole recording",
}


def _field(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("FieldLabel")
    return label


def _row(*widgets, stretch=True) -> QWidget:
    w = QWidget()
    row = QHBoxLayout(w)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(8)
    for x in widgets:
        if x is None:
            row.addSpacing(12)
        else:
            row.addWidget(x)
    if stretch:
        row.addStretch(1)
    return w


class OptionsPanel(Card):
    changed = Signal()
    download_whisper = Signal(str)
    add_llm = Signal()

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent, margins=20, spacing=0)
        self.whisper = ModelPicker("No Whisper model yet")
        self.llm = ModelPicker("No LLM yet: add a .gguf model")
        self.preset = SegmentedControl([(k, label.split(" /")[0]) for k, (label, _, _) in PRESETS.items()]
                                       + [("custom", "Custom")])
        self.chips = {key: Chip(label, OUTPUT_TIPS[key]) for key, label in OUTPUTS.items()}
        self.speakers = Chip("Detect speakers", "Label who is speaking (Speaker 1, Speaker 2, ...). Runs on the CPU.")
        self.tone = QComboBox()
        for key, label in STORY_TONES.items():
            self.tone.addItem(label, key)
        self.minutes = QComboBox()
        for m in STORY_MINUTES:
            self.minutes.addItem(f"{m} min read", m)
        # One chip per translation language (Tamil today); at most one is on.
        self.translate = {code: Chip(f"Translate to {name}",
                                     f"Also write the story in {name} once it is finished. Runs offline with "
                                     f"{translator.MODEL_NAME} (1.3 GB download on first use).")
                          for code, (name, _, _) in translator.LANGUAGES.items()}
        self.reading = QLabel()
        self.reading.setObjectName("Reading")

        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(12)
        grid.setColumnStretch(1, 1)
        self.story_label = _field("Story")
        self.story_row = _row(self.tone, self.minutes, None, *self.translate.values())
        rows = [
            ("Transcribe model", _row(self.whisper)),
            ("LLM model", _row(self.llm)),
            ("Content", self.preset),
            ("Output", _row(*self.chips.values(), None, self.speakers)),
            (self.story_label, self.story_row),
            ("Reading time", _row(icon_label(theme.ICON_CLOCK, 10), self.reading)),
        ]
        for i, (label, widget) in enumerate(rows):
            grid.addWidget(label if isinstance(label, QLabel) else _field(label), i, 0, Qt.AlignVCenter)
            grid.addWidget(widget, i, 1)
        self.body.addLayout(grid)

        self.load(settings)
        self.refresh_models(settings)
        self.whisper.selected.connect(lambda _: self.changed.emit())
        self.llm.selected.connect(lambda _: self.changed.emit())
        self.preset.changed.connect(self._preset_chosen)
        for chip in [*self.chips.values(), self.speakers]:
            chip.toggled.connect(self._chips_changed)
        self.tone.currentIndexChanged.connect(self.changed)
        self.minutes.currentIndexChanged.connect(self.changed)
        for code, chip in self.translate.items():
            chip.toggled.connect(lambda on, code=code: self._translate_toggled(code, on))

    def _translate_toggled(self, code: str, on: bool) -> None:
        if on:  # one language at a time
            for other, chip in self.translate.items():
                if other != code and chip.isChecked():
                    chip.blockSignals(True)
                    chip.setChecked(False)
                    chip.blockSignals(False)
        self.changed.emit()

    # ------------------------------------------------------------------ models

    def refresh_models(self, s: Settings) -> None:
        installed = models.installed_whisper()
        if s.whisper_model not in {m.key for m in installed} and s.whisper_model in models.WHISPER_CATALOG:
            mb = models.WHISPER_CATALOG[s.whisper_model][1]
            installed.append(models.ModelInfo(s.whisper_model, s.whisper_model, mb * 1_000_000,
                                              "downloads on first run"))
        downloads = [(f"{m.label}  ·  {m.note}\t{m.size}", lambda key=m.key: self.download_whisper.emit(key))
                     for m in models.downloadable_whisper() if m.key != s.whisper_model]
        self.whisper.setEnabled(True)
        self.whisper.set_models(installed, s.whisper_model, [("Download another model", downloads)])
        self.llm.set_models(models.installed_llms(s.llm_path), s.llm_path, [
            ("Add a model file…", self.add_llm.emit),
            ("Open models folder", lambda: models.open_folder(models_dir())),
        ])

    # ------------------------------------------------------------------ settings

    def load(self, s: Settings) -> None:
        widgets = [self.speakers, self.tone, self.minutes, *self.chips.values(), *self.translate.values()]
        for w in widgets:
            w.blockSignals(True)
        for code, chip in self.translate.items():
            chip.setChecked(s.translate_to == code)
        self.preset.set_current(s.content_type if s.content_type in self.preset.buttons else "custom")
        for key, chip in self.chips.items():
            chip.setChecked(key in s.outputs)
        self.speakers.setChecked(s.diarize)
        self.tone.setCurrentIndex(max(0, self.tone.findData(s.story_tone)))
        index = self.minutes.findData(s.story_minutes)
        if index < 0:  # a custom value (e.g. from the CLI) still shows
            self.minutes.addItem(f"{s.story_minutes} min read", s.story_minutes)
            index = self.minutes.count() - 1
        self.minutes.setCurrentIndex(index)
        for w in widgets:
            w.blockSignals(False)
        self._story_visible("story" in s.outputs)

    def apply_to(self, s: Settings) -> None:
        s.content_type = self.preset.current() or "custom"
        s.outputs = [k for k, chip in self.chips.items() if chip.isChecked()]
        s.diarize = self.speakers.isChecked()
        s.story_tone = self.tone.currentData()
        s.story_minutes = self.minutes.currentData()
        s.translate_to = next((code for code, chip in self.translate.items() if chip.isChecked()), "")
        if self.whisper.current:
            s.whisper_model = self.whisper.current
        s.llm_path = self.llm.current

    def set_audio_available(self, available: bool) -> None:
        """Speaker detection needs audio; subtitle files bring their own names."""
        self.speakers.setEnabled(available)
        self.speakers.setToolTip(
            "Label who is speaking (Speaker 1, Speaker 2, ...). Runs on the CPU." if available else
            "Not available for subtitle files: speaker names written in the subtitles are used.")

    def set_reading(self, text: str) -> None:
        self.reading.setText(text)

    def _story_visible(self, visible: bool) -> None:
        self.story_label.setVisible(visible)
        self.story_row.setVisible(visible)

    def _preset_chosen(self, key: str) -> None:
        if key in PRESETS:
            s = Settings()
            self.apply_to(s)
            s.apply_preset(key)
            self.load(s)
        self.changed.emit()

    def _chips_changed(self) -> None:
        s = Settings()
        self.apply_to(s)
        match = next((k for k, (_, outs, diar) in PRESETS.items()
                      if outs == s.outputs and diar == s.diarize), "custom")
        self.preset.set_current(match)
        self._story_visible("story" in s.outputs)
        self.changed.emit()
