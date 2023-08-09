import numpy

from PySide6.QtWidgets import QWidget, QDialog, QLabel, QVBoxLayout
from PySide6.QtGui import QImage, QPixmap, QPainter, QPolygon, QPen, QColor, QBrush, QPaintEvent
from PySide6.QtCore import Qt, Signal


class ImgDialog(QDialog):
    def __init__(self, parent=None, title="", modality=Qt.WindowModality.NonModal, polygon=False):
        super().__init__(parent=parent)

        # self.img = QImage()
        layout = QVBoxLayout()
        self.img_label = PolygonOverlayLabel() if polygon is True else QLabel()
        self.img_pixmap = QPixmap()
        layout.addWidget(self.img_label)

        self.setLayout(layout)
        self.setWindowTitle(title)
        self.setWindowModality(modality)

    def set_image(self, img, scale=False):
        self.img_label.setPixmap(self.img_pixmap.fromImage(img))
        self.img_label.setScaledContents(scale)

    def set_array(self, arr, scale=False):
        img = QImage(arr.data, arr.shape[1], arr.shape[0], QImage.Format.Format_BGR888)
        self.set_image(img)
        self.img_label.setScaledContents(scale)

    def set_file(self, path, scale=False):
        self.img_pixmap.load(path)
        self.img_label.setPixmap(self.img_pixmap)
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setScaledContents(scale)


class ImgWidget(QWidget):
    def __init__(self, parent=None, polygon=False):
        super().__init__(parent=parent)

        # self.img = QImage()
        layout = QVBoxLayout()
        self.img_label = PolygonOverlayLabel() if polygon is True else QLabel()
        self.img_pixmap = QPixmap()
        layout.addWidget(self.img_label)

        self.setLayout(layout)

    def set_image(self, img, scale=False):
        self.img_label.setPixmap(self.img_pixmap.fromImage(img))
        self.img_label.setScaledContents(scale)

    def set_array(self, arr, scale=False):
        img = QImage(arr.data, arr.shape[1], arr.shape[0], QImage.Format.Format_BGR888)
        self.set_image(img)
        self.img_label.setScaledContents(scale)

    def set_file(self, path, scale=False):
        self.img_pixmap.load(path)
        self.img_label.setPixmap(self.img_pixmap)
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setScaledContents(scale)


class PolygonOverlayLabel(QLabel):
    draw = Signal(numpy.ndarray, bool)

    def __init__(self):
        super().__init__()

        self.polygon_points = []

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)

        qp = QPainter()
        qp.begin(self)
        for polygon_p in self.polygon_points:
            self.draw_polygon(qp, polygon_p)
            # self.setScaledContents(True)
        qp.end()

    def draw_polygon(self, qp, points):
        polygon = QPolygon(points)

        # line
        qp.setPen(QPen(QColor(96, 96, 255), 3))

        # color
        qp.setBrush(QBrush(QColor(196, 196, 255, 96)))

        # draw
        qp.drawPolygon(polygon)
