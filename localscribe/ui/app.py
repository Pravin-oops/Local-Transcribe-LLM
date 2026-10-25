import sys

from PySide6.QtWidgets import QApplication

from localscribe.ui.main_window import MainWindow


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("LocalScribe")
    window = MainWindow()
    window.show()
    return app.exec()
