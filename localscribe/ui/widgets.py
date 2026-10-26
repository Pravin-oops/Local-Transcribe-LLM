"""Reusable components: card, drop zone, Ollama-style model picker, segmented
control and toggle chips. Styling lives in theme.py; these only set object
names and dynamic properties."""
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QButtonGroup, QFrame, QHBoxLayout, QLabel, QMenu, QPushButton, QVBoxLayout,
                               QWidget)

from localscribe.ui import theme


class Card(QFrame):
    def __init__(self, parent=None, margins=18, spacing=12):
        super().__init__(parent)
        self.setObjectName("Card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(margins, margins, margins, margins)
        self.body.setSpacing(spacing)


def icon_label(glyph: str, size: int = 12) -> QLabel:
    label = QLabel(glyph)
    label.setObjectName("Icon")
    label.setStyleSheet(f"font-size: {size}pt;")
    return label


class DropZone(QFrame):
    """Large rounded drop target that also opens the file dialog when clicked."""
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DropZone")
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(118)
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(20, 16, 20, 16)
        self.icon = QLabel(theme.ICON_UPLOAD)
        self.icon.setObjectName("DropIcon")
        self.title = QLabel("Drag and drop or click to browse")
        self.title.setObjectName("DropTitle")
        self.detail = QLabel("Audio, video, or .srt / .vtt subtitles")
        self.detail.setObjectName("Muted")
        for w in (self.icon, self.title, self.detail):
            w.setAlignment(Qt.AlignCenter)
            w.setAttribute(Qt.WA_TransparentForMouseEvents)
            layout.addWidget(w)

    def set_active(self, active: bool) -> None:
        self.setProperty("active", active)
        theme.repolish(self)

    def set_file(self, name: str, detail: str) -> None:
        self.icon.setText(theme.ICON_DOCUMENT)
        self.title.setText(name)
        self.detail.setText(detail)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()

    def enterEvent(self, event):
        self.set_active(True)

    def leaveEvent(self, event):
        self.set_active(False)


class ModelPicker(QPushButton):
    """Pill showing the current model; opens a rounded menu of installed models
    (name and size), followed by extra actions such as 'Add a model…'."""
    selected = Signal(str)

    def __init__(self, empty_text: str, parent=None):
        super().__init__(parent)
        self.setObjectName("Pill")
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumWidth(320)
        self.empty_text = empty_text
        self.models, self.current, self.extras = [], "", []
        row = QHBoxLayout(self)
        row.setContentsMargins(14, 0, 12, 0)
        row.setSpacing(8)
        self.name = QLabel()
        self.size = QLabel()
        self.size.setObjectName("Muted")
        self.chevron = icon_label(theme.ICON_CHEVRON, 8)
        for w in (self.name, self.size, self.chevron):
            w.setAttribute(Qt.WA_TransparentForMouseEvents)
        row.addWidget(self.name)
        row.addStretch(1)
        row.addWidget(self.size)
        row.addWidget(self.chevron)
        self.clicked.connect(self._open_menu)

    def set_models(self, models: list, current: str, extras: list = ()) -> None:
        """``extras``: (text, callback) actions shown under a separator; a list
        value instead of a callback makes a submenu of (text, callback)."""
        self.models, self.extras = list(models), list(extras)
        keys = [m.key for m in self.models]
        self.current = current if current in keys else (keys[0] if keys else "")
        info = next((m for m in self.models if m.key == self.current), None)
        self.name.setText(info.label if info else self.empty_text)
        self.size.setText(info.size if info else "")
        self.setToolTip(info.note if info and info.note else "")

    def set_busy(self, text: str) -> None:
        self.name.setText(text)
        self.size.setText("")
        self.setEnabled(False)

    def build_menu(self) -> QMenu:
        menu = QMenu(self)
        menu.setAttribute(Qt.WA_TranslucentBackground)
        menu.setWindowFlags(menu.windowFlags() | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        menu.setMinimumWidth(self.width())
        for m in self.models:
            note = f"  ·  {m.note}" if m.note else ""
            action = QAction(f"{m.label}{note}\t{m.size}", menu)  # text after \t is right-aligned
            action.setCheckable(True)
            action.setChecked(m.key == self.current)
            action.triggered.connect(lambda _=False, key=m.key: self._choose(key))
            menu.addAction(action)
        if not self.models:
            menu.addAction(self.empty_text).setEnabled(False)
        if self.extras:
            menu.addSeparator()
        for text, target in self.extras:
            if isinstance(target, list):
                sub = menu.addMenu(text)
                sub.setAttribute(Qt.WA_TranslucentBackground)
                sub.setWindowFlags(sub.windowFlags() | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
                for sub_text, callback in target:
                    sub.addAction(sub_text).triggered.connect(lambda _=False, cb=callback: cb())
                if not target:
                    sub.addAction("All models are already downloaded").setEnabled(False)
            else:
                menu.addAction(text).triggered.connect(lambda _=False, cb=target: cb())
        return menu

    def _open_menu(self):
        self.build_menu().exec(self.mapToGlobal(QPoint(0, self.height() + 4)))

    def _choose(self, key: str):
        self.set_models(self.models, key, self.extras)
        self.selected.emit(key)


class SegmentedControl(QWidget):
    """Joined pill buttons, one checked at a time."""
    changed = Signal(str)

    def __init__(self, options: list, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons = {}
        for i, (key, label) in enumerate(options):
            b = QPushButton(label)
            b.setObjectName("Segment")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            # Not "pos": that name is QWidget's own position property.
            b.setProperty("segment", "first" if i == 0 else "last" if i == len(options) - 1 else "mid")
            self.group.addButton(b)
            self.buttons[key] = b
            row.addWidget(b)
            b.clicked.connect(lambda _=False, k=key: self.changed.emit(k))
        row.addStretch(1)

    def current(self) -> str:
        return next((k for k, b in self.buttons.items() if b.isChecked()), "")

    def set_current(self, key: str) -> None:
        if key in self.buttons:
            self.buttons[key].setChecked(True)


class Chip(QPushButton):
    """Toggle chip: a rounded checkable button."""

    def __init__(self, text: str, tooltip: str = "", parent=None):
        super().__init__(text, parent)
        self.setObjectName("Chip")
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(tooltip)
