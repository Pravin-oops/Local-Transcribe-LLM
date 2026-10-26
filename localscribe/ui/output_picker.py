"""Chooses what a job produces: a content-type preset plus per-output checkboxes.

Picking a preset ticks its outputs; ticking boxes by hand switches the preset
to the matching one, or to Custom. Only ticked outputs are computed.
"""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from localscribe.config import OUTPUTS, PRESETS, STORY_LENGTHS, STORY_TONES, Settings

OUTPUT_TIPS = {
    "summary": "Overview, key points and timestamped chapters",
    "notes": "Key concepts, takeaways, examples and review questions",
    "story": "A captivating narrator script that retells the whole recording",
}


class OutputPicker(QGroupBox):
    changed = Signal()

    def __init__(self, settings: Settings, parent=None):
        super().__init__("Output", parent)
        self.preset = QComboBox()
        for key, (label, _, _) in PRESETS.items():
            self.preset.addItem(label, key)
        self.preset.addItem("Custom", "custom")
        self.boxes = {}
        for key, label in OUTPUTS.items():
            box = QCheckBox(label)
            box.setToolTip(OUTPUT_TIPS[key])
            self.boxes[key] = box
        self.speakers = QCheckBox("Detect speakers")
        self.speakers.setToolTip("Label who is speaking (Speaker 1, Speaker 2, ...). Runs on the CPU.")
        self.tone = QComboBox()
        for key, label in STORY_TONES.items():
            self.tone.addItem(label, key)
        self.length = QComboBox()
        for key, (label, _) in STORY_LENGTHS.items():
            self.length.addItem(label, key)

        top = QHBoxLayout()
        top.addWidget(QLabel("Content:"))
        top.addWidget(self.preset)
        top.addSpacing(12)
        for box in self.boxes.values():
            top.addWidget(box)
        top.addSpacing(12)
        top.addWidget(self.speakers)
        top.addStretch(1)

        self.story_row = QWidget()
        story = QHBoxLayout(self.story_row)
        story.setContentsMargins(0, 0, 0, 0)
        story.addWidget(QLabel("Story narration:"))
        story.addWidget(self.tone)
        story.addWidget(QLabel("Length:"))
        story.addWidget(self.length)
        story.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.story_row)

        self.load(settings)
        self.preset.currentIndexChanged.connect(self._preset_chosen)
        for box in [*self.boxes.values(), self.speakers]:
            box.toggled.connect(self._boxes_changed)
        self.tone.currentIndexChanged.connect(self.changed)
        self.length.currentIndexChanged.connect(self.changed)

    def load(self, s: Settings) -> None:
        widgets = [self.preset, self.speakers, self.tone, self.length, *self.boxes.values()]
        for w in widgets:
            w.blockSignals(True)
        self.preset.setCurrentIndex(max(0, self.preset.findData(s.content_type)))
        for key, box in self.boxes.items():
            box.setChecked(key in s.outputs)
        self.speakers.setChecked(s.diarize)
        self.tone.setCurrentIndex(max(0, self.tone.findData(s.story_tone)))
        self.length.setCurrentIndex(max(0, self.length.findData(s.story_length)))
        for w in widgets:
            w.blockSignals(False)
        self.story_row.setVisible("story" in s.outputs)

    def apply_to(self, s: Settings) -> None:
        s.content_type = self.preset.currentData()
        s.outputs = [k for k, box in self.boxes.items() if box.isChecked()]
        s.diarize = self.speakers.isChecked()
        s.story_tone = self.tone.currentData()
        s.story_length = self.length.currentData()

    def _preset_chosen(self) -> None:
        key = self.preset.currentData()
        if key in PRESETS:
            s = Settings()
            self.apply_to(s)
            s.apply_preset(key)
            self.load(s)
        self.changed.emit()

    def _boxes_changed(self) -> None:
        s = Settings()
        self.apply_to(s)
        match = next((k for k, (_, outs, diar) in PRESETS.items()
                      if outs == s.outputs and diar == s.diarize), "custom")
        self.preset.blockSignals(True)
        self.preset.setCurrentIndex(self.preset.findData(match))
        self.preset.blockSignals(False)
        self.story_row.setVisible("story" in s.outputs)
        self.changed.emit()
