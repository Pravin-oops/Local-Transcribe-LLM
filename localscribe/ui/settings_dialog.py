"""Advanced settings. Models, outputs and story length live on the main page."""
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from localscribe.config import Settings


def _path_row(edit: QLineEdit, on_browse) -> QWidget:
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(edit, 1)
    button = QPushButton("Browse…")
    button.clicked.connect(on_browse)
    layout.addWidget(button)
    return row


def _section(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet("font-weight: 600; padding-top: 10px;")
    return label


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(560)
        self._settings = settings

        self.device = QComboBox()
        self.device.addItems(["auto", "cuda", "cpu"])
        self.device.setCurrentText(settings.device)

        self.compute_type = QComboBox()
        self.compute_type.addItems(["auto", "int8_float16", "float16", "int8", "float32"])
        self.compute_type.setCurrentText(settings.compute_type)

        self.beam_size = QSpinBox()
        self.beam_size.setRange(1, 10)
        self.beam_size.setValue(settings.beam_size)

        self.batch_size = QSpinBox()
        self.batch_size.setRange(0, 32)
        self.batch_size.setSpecialValueText("Off")
        self.batch_size.setValue(settings.batch_size)
        self.batch_size.setToolTip("Batched decoding is 2-4x faster on GPU but uses more VRAM.")

        self.chunk_minutes = QDoubleSpinBox()
        self.chunk_minutes.setRange(1, 30)
        self.chunk_minutes.setSingleStep(1)
        self.chunk_minutes.setSuffix(" min")
        self.chunk_minutes.setValue(settings.chunk_minutes)

        self.num_speakers = QSpinBox()
        self.num_speakers.setRange(0, 20)
        self.num_speakers.setSpecialValueText("Detect automatically")
        self.num_speakers.setValue(settings.num_speakers)
        self.num_speakers.setToolTip("Leave on automatic. Set a number only if people are merged or split.")

        self.llm_ctx = QComboBox()
        self.llm_ctx.addItems(["4096", "8192", "16384"])
        self.llm_ctx.setCurrentText(str(settings.llm_ctx))

        self.llm_gpu_layers = QSpinBox()
        self.llm_gpu_layers.setRange(-1, 200)
        self.llm_gpu_layers.setSpecialValueText("Auto (fit to free VRAM)")
        self.llm_gpu_layers.setValue(settings.llm_gpu_layers)

        self.output_dir = QLineEdit(settings.output_dir)

        form = QFormLayout()
        form.setVerticalSpacing(10)
        form.setHorizontalSpacing(18)
        form.addRow(_section("Transcription (English)"))
        form.addRow("Device", self.device)
        form.addRow("Precision", self.compute_type)
        form.addRow("Beam size", self.beam_size)
        form.addRow("Batch size", self.batch_size)
        form.addRow("Chunk length", self.chunk_minutes)
        form.addRow(_section("Speakers"))
        form.addRow("Number of speakers", self.num_speakers)
        form.addRow(_section("LLM"))
        form.addRow("Context size", self.llm_ctx)
        form.addRow("GPU layers", self.llm_gpu_layers)
        form.addRow(_section("Output"))
        form.addRow("Folder", _path_row(self.output_dir, self._browse_output))

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 14, 22, 18)
        layout.addLayout(form)
        layout.addSpacing(8)
        layout.addWidget(buttons)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Choose output folder", self.output_dir.text())
        if path:
            self.output_dir.setText(path)

    def result_settings(self) -> Settings:
        data = self._settings.to_dict()
        data.update(
            device=self.device.currentText(),
            compute_type=self.compute_type.currentText(),
            beam_size=self.beam_size.value(),
            batch_size=self.batch_size.value(),
            chunk_minutes=self.chunk_minutes.value(),
            num_speakers=self.num_speakers.value(),
            llm_ctx=int(self.llm_ctx.currentText()),
            llm_gpu_layers=self.llm_gpu_layers.value(),
            output_dir=self.output_dir.text().strip() or self._settings.output_dir,
        )
        return Settings.from_dict(data)
