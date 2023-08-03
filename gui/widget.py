from PySide6.QtWidgets import QWidget, QDialog, QLabel, QVBoxLayout
from PySide6.QtCore import Qt

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
