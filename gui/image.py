
from PySide6.QtWidgets import QWidget, QDialog, QLabel, QVBoxLayout
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt


class ImgDialog(QDialog):
    def __init__(self, parent=None, title="", modality=Qt.WindowModality.NonModal):
        super().__init__(parent=parent)

        # self.img = QImage()
        layout = QVBoxLayout()
        self.img_label = QLabel()
        self.img_pixmap = QPixmap()
        layout.addWidget(self.img_label)

        self.setLayout(layout)
        self.setWindowTitle(title)
        self.setWindowModality(modality)

    def set_image(self, img):
        self.img_label.setPixmap(self.img_pixmap.fromImage(img))

    def set_file(self, path):
        self.img_pixmap.load(path)
        self.img_label.setPixmap(self.img_pixmap)
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)


class ImgWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        # self.img = QImage()
        layout = QVBoxLayout()
        self.img_label = QLabel()
        self.img_pixmap = QPixmap()
        layout.addWidget(self.img_label)

        self.setLayout(layout)

    def set_image(self, img):
        self.img_label.setPixmap(self.img_pixmap.fromImage(img))

    def set_file(self, path):
        self.img_pixmap.load(path)
        self.img_label.setPixmap(self.img_pixmap)
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)