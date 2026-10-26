import sys

from PySide6.QtWidgets import QApplication

from localscribe.ui import theme
from localscribe.ui.main_window import MainWindow


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("LocalScribe")
    theme.apply(app)
    window = MainWindow()
    window.show()
    return app.exec()
