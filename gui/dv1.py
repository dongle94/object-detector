import os
import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtCore import QSize, Slot


class MainWidget(QWidget):
    def __init__(self):
        super().__init__()


class DaolCND(QMainWindow):
    def __init__(self):
        super().__init__()
        self.geo = self.screen().availableGeometry()
        self.resize(QSize(self.geo.width() * 0.5, self.geo.height() * 0.5))

        # MenuBar
        self.mbar = self.menuBar()
        self.file_menu = self.mbar.addMenu("File")

        # Menu - Exit QAction
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        self.file_menu.addAction(exit_action)

        # Status Bar
        self.statusBar().showMessage("ready state", 0)

        # Set Central Widget
        self.main_widget = MainWidget()
        self.setCentralWidget(self.main_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    app_gui = DaolCND()
    app_gui.setWindowTitle("다올씨앤디")
    app_gui.show()

    sys.exit(app.exec())
