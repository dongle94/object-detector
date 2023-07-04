import os
import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
from PySide6.QtCore import QSize

class DaolCND(QMainWindow):
    def __init__(self):
        super().__init__()
        self.geo = self.screen().availableGeometry()
        self.resize(QSize(self.geo.width() * 0.5, self.geo.height() * 0.5))

        self.mbar = self.menuBar()
        self.mbar.addMenu("File")

        self.statusBar().showMessage("ready state", 0)


class MainWidget(QWidget):
    def __init__(self):
        super().__init__()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    app_gui = DaolCND()
    app_gui.setWindowTitle("다올씨앤디")
    app_gui.show()

    sys.exit(app.exec())
