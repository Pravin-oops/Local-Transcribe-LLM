"""Light/dark theme in a calm, Ollama-like style: neutral surfaces, rounded
cards and pills, one accent. Follows the Windows colour scheme and switches
live when it changes."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPalette
from PySide6.QtWidgets import QApplication

ICON_FONT = "Segoe Fluent Icons"
UI_FONT = "Segoe UI Variable"

# Segoe Fluent Icons code points
ICON_UPLOAD = ""
ICON_SETTINGS = ""
ICON_CHEVRON = ""
ICON_FOLDER = ""
ICON_ADD = ""
ICON_DOWNLOAD = ""
ICON_CHECK = ""
ICON_DOCUMENT = ""
ICON_CLOCK = ""

LIGHT = dict(bg="#f6f6f7", surface="#ffffff", surface2="#f0f0f2", hover="#e8e8eb", border="#e3e3e7",
             text="#18181b", muted="#71717a", accent="#2563eb", accent_soft="#e8efff", on_strong="#ffffff")
DARK = dict(bg="#151517", surface="#1e1e21", surface2="#29292d", hover="#333338", border="#323237",
            text="#f4f4f5", muted="#a1a1aa", accent="#60a5fa", accent_soft="#1c2a42", on_strong="#151517")


def is_dark() -> bool:
    return QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark


def colors() -> dict:
    return DARK if is_dark() else LIGHT


def stylesheet(c: dict) -> str:
    return f"""
* {{ font-family: "{UI_FONT}", "Segoe UI"; font-size: 10pt; color: {c['text']}; }}
QMainWindow, QDialog {{ background: {c['bg']}; }}
QWidget#Page {{ background: {c['bg']}; }}
QFrame#Card {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 14px; }}
QLabel {{ background: transparent; }}
QLabel#Title {{ font-size: 16pt; font-weight: 600; }}
QLabel#Muted, QLabel#FieldLabel {{ color: {c['muted']}; }}
QLabel#Icon {{ font-family: "{ICON_FONT}"; color: {c['muted']}; }}
QLabel#Reading {{ color: {c['muted']}; }}

QFrame#DropZone {{ background: {c['surface']}; border: 2px dashed {c['border']}; border-radius: 16px; }}
QFrame#DropZone[active="true"] {{ border-color: {c['accent']}; background: {c['accent_soft']}; }}
QLabel#DropIcon {{ font-family: "{ICON_FONT}"; font-size: 22pt; color: {c['muted']}; }}
QFrame#DropZone[active="true"] QLabel#DropIcon {{ color: {c['accent']}; }}
QLabel#DropTitle {{ font-size: 12pt; font-weight: 600; }}

QPushButton {{ background: {c['surface2']}; border: 1px solid {c['border']}; border-radius: 10px;
              padding: 7px 14px; }}
QPushButton:hover {{ background: {c['hover']}; }}
QPushButton:disabled {{ color: {c['muted']}; }}
QPushButton#Primary {{ background: {c['text']}; color: {c['on_strong']}; border: none; border-radius: 18px;
                      padding: 9px 26px; font-weight: 600; }}
QPushButton#Primary:hover {{ background: {c['muted']}; }}
QPushButton#Primary:disabled {{ background: {c['hover']}; color: {c['muted']}; }}
QPushButton#Ghost {{ background: transparent; border: none; color: {c['muted']}; padding: 6px 10px; }}
QPushButton#Ghost:hover {{ color: {c['text']}; background: {c['surface2']}; }}
QPushButton#IconButton {{ font-family: "{ICON_FONT}"; font-size: 12pt; background: transparent; border: none;
                         border-radius: 16px; padding: 6px; color: {c['muted']}; }}
QPushButton#IconButton:hover {{ background: {c['surface2']}; color: {c['text']}; }}

QPushButton#Pill {{ border-radius: 17px; padding: 0px 12px; min-height: 34px; text-align: left; }}
QPushButton#Chip {{ border-radius: 16px; padding: 6px 14px; }}
QPushButton#Chip:checked {{ background: {c['text']}; color: {c['on_strong']}; border-color: {c['text']}; }}
QPushButton#Chip:checked:disabled {{ background: {c['hover']}; color: {c['muted']}; border-color: {c['border']}; }}
QPushButton#Segment {{ border-radius: 0px; padding: 6px 16px; margin: 0px; }}
QPushButton#Segment[segment="first"] {{ border-top-left-radius: 16px; border-bottom-left-radius: 16px; }}
QPushButton#Segment[segment="last"] {{ border-top-right-radius: 16px; border-bottom-right-radius: 16px; }}
QPushButton#Segment:checked {{ background: {c['text']}; color: {c['on_strong']}; border-color: {c['text']}; }}

QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{ background: {c['surface2']}; border: 1px solid {c['border']};
    border-radius: 10px; padding: 6px 10px; min-height: 20px; }}
QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ background: {c['hover']}; }}
QComboBox {{ padding-right: 28px; }}
QComboBox::drop-down {{ border: none; width: 26px; }}
QComboBox::down-arrow {{ image: url({c['arrow_down']}); width: 10px; height: 10px; }}
QSpinBox, QDoubleSpinBox {{ padding-right: 22px; }}
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {{
    border: none; background: transparent; width: 20px; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({c['arrow_up']}); width: 9px; height: 9px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({c['arrow_down']}); width: 9px; height: 9px; }}
QComboBox QAbstractItemView {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 10px;
    padding: 4px; outline: 0; selection-background-color: {c['surface2']}; selection-color: {c['text']}; }}

QMenu {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 12px; padding: 6px; }}
QMenu::item {{ padding: 8px 18px 8px 30px; border-radius: 8px; }}
QMenu::item:selected {{ background: {c['surface2']}; }}
QMenu::item:disabled {{ color: {c['muted']}; }}
QMenu::separator {{ height: 1px; background: {c['border']}; margin: 6px 8px; }}
QMenu::indicator {{ left: 8px; width: 14px; height: 14px; }}

QProgressBar {{ background: {c['surface2']}; border: none; border-radius: 3px; max-height: 6px; }}
QProgressBar::chunk {{ background: {c['accent']}; border-radius: 3px; }}

QTabWidget::pane {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 14px; top: 6px; }}
QTabBar::tab {{ background: transparent; color: {c['muted']}; padding: 7px 14px; margin-right: 4px;
               border-radius: 14px; }}
QTabBar::tab:selected {{ background: {c['surface2']}; color: {c['text']}; }}
QTabBar::tab:hover:!selected {{ color: {c['text']}; }}
QPlainTextEdit, QTextBrowser {{ background: {c['surface']}; border: none; border-radius: 12px; padding: 10px;
    selection-background-color: {c['accent_soft']}; selection-color: {c['text']}; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 4px 2px; }}
QScrollBar::handle:vertical {{ background: {c['border']}; border-radius: 3px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {c['muted']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0px; width: 0px; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QToolTip {{ background: {c['surface']}; color: {c['text']}; border: 1px solid {c['border']};
           border-radius: 8px; padding: 6px; }}
QCheckBox {{ spacing: 8px; }}
"""


def _arrow_images(c: dict) -> dict:
    """Chevron PNGs in the theme's muted colour (QSS needs image files for arrows)."""
    from PySide6.QtGui import QPainter, QPixmap

    from localscribe.config import app_data_dir
    folder = app_data_dir() / "ui"
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, glyph in (("arrow_down", ICON_CHEVRON), ("arrow_up", "")):
        path = folder / f"{name}-{c['muted'].lstrip('#')}.png"
        if not path.exists():
            pix = QPixmap(40, 40)
            pix.fill(Qt.transparent)
            painter = QPainter(pix)
            painter.setPen(QColor(c["muted"]))
            painter.setFont(QFont(ICON_FONT, 20))
            painter.drawText(pix.rect(), Qt.AlignCenter, glyph)
            painter.end()
            pix.save(str(path))
        paths[name] = path.as_posix()
    return paths


def apply(app: QApplication) -> None:
    """Apply the theme now and re-apply whenever Windows switches light/dark."""
    app.setStyle("Fusion")  # QSS renders consistently on Fusion
    app.setFont(QFont(UI_FONT, 10))

    def refresh(*_):
        c = dict(colors())
        c.update(_arrow_images(c))
        pal = QPalette()
        for role, key in ((QPalette.Window, "bg"), (QPalette.Base, "surface"), (QPalette.AlternateBase, "surface2"),
                          (QPalette.Button, "surface2"), (QPalette.Text, "text"), (QPalette.WindowText, "text"),
                          (QPalette.ButtonText, "text"), (QPalette.Highlight, "accent"),
                          (QPalette.PlaceholderText, "muted"), (QPalette.ToolTipBase, "surface"),
                          (QPalette.ToolTipText, "text")):
            pal.setColor(role, QColor(c[key]))
        pal.setColor(QPalette.HighlightedText, QColor(c["on_strong"]))
        app.setPalette(pal)
        app.setStyleSheet(stylesheet(c))

    refresh()
    QGuiApplication.styleHints().colorSchemeChanged.connect(refresh)


def repolish(widget) -> None:
    """Re-evaluate QSS after changing a dynamic property used in selectors."""
    from PySide6.QtWidgets import QWidget
    for w in [widget, *widget.findChildren(QWidget)]:
        w.style().unpolish(w)
        w.style().polish(w)
