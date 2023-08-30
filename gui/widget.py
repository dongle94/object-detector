import numpy

from PySide6.QtWidgets import QWidget, QDialog, QLabel, QVBoxLayout
from PySide6.QtCore import Qt, Signal

class MsgDialog(QDialog):
    def __init__(self, parent=None, msg="", title="", modality=Qt.WindowModality.NonModal):
        super().__init__(parent=parent)

        layout = QVBoxLayout()
        self.label = QLabel(msg)
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.setWindowTitle(title)
        self.setWindowModality(modality)
        self.show()

class NormalLabel(QLabel):
    draw = Signal(numpy.ndarray, bool)
    updateText = Signal(str)

    def __init__(self, *args):
        super().__init__(*args)
