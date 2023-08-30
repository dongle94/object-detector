import numpy

from PySide6.QtWidgets import QWidget, QDialog, QLabel, QVBoxLayout
from PySide6.QtGui import QImage, QPixmap, QPainter, QPolygon, QPen, QColor, QBrush, QPaintEvent
from PySide6.QtCore import Qt, Signal


class ImgDialog(QDialog):
    def __init__(self, parent=None, title="", modality=Qt.WindowModality.NonModal, polygon=False):
        super().__init__(parent=parent)

        # self.img = QImage()
        layout = QVBoxLayout()
        self.img_label = PolygonOverlayLabel() if polygon is True else NormalLabel()
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
        self.img_label = PolygonOverlayLabel() if polygon is True else NormalLabel()
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


class NormalLabel(QLabel):
    draw = Signal(numpy.ndarray, bool)

    def __init__(self):
        super().__init__()


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


class EllipseLabel(QLabel):
    updateFillColor = Signal(tuple)
    updateLineColor = Signal(tuple)
    updateSize = Signal(tuple)

    def __init__(self, size, line_color=(0, 0, 0), fill_color=(128, 128, 128)):
        super().__init__()

        self.draw_size = size
        self.line_color = line_color
        self.fill_color = fill_color

    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        self.draw_ellipse(qp)
        qp.end()

    def draw_ellipse(self, qp):
        qp.setPen(QPen(QColor(*self.line_color), 1))
        qp.setBrush(QBrush(QColor(*self.fill_color), Qt.SolidPattern))
        qp.drawEllipse(*self.draw_size)


