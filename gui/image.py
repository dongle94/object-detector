
from PySide6.QtWidgets import QWidget, QDialog, QLabel, QVBoxLayout
from PySide6.QtGui import QImage, QPixmap, QPainter, QPolygon, QPen, QColor, QBrush, QPaintEvent
from PySide6.QtCore import Qt


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

    def set_image(self, img):
        self.img_label.setPixmap(self.img_pixmap.fromImage(img))

    def set_array(self, arr):
        img = QImage(arr.data, arr.shape[1], arr.shape[0], QImage.Format.Format_BGR888)
        self.set_image(img)

    def set_file(self, path):
        self.img_pixmap.load(path)
        self.img_label.setPixmap(self.img_pixmap)
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)


class ImgWidget(QWidget):
    def __init__(self, parent=None, polygon=False):
        super().__init__(parent=parent)

        # self.img = QImage()
        layout = QVBoxLayout()
        self.img_label = PolygonOverlayLabel() if polygon is True else QLabel()
        self.img_pixmap = QPixmap()
        layout.addWidget(self.img_label)

        self.setLayout(layout)

    def set_image(self, img):
        self.img_label.setPixmap(self.img_pixmap.fromImage(img))

    def set_array(self, arr):
        img = QImage(arr.data, arr.shape[1], arr.shape[0], QImage.Format.Format_BGR888)
        self.set_image(img)

    def set_file(self, path):
        self.img_pixmap.load(path)
        self.img_label.setPixmap(self.img_pixmap)
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)


class PolygonOverlayLabel(QLabel):
    def __init__(self):
        super().__init__()

        self.polygon_points = []

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)

        qp = QPainter()
        qp.begin(self)
        if len(self.polygon_points) >= 3:
            self.darw_polygon(qp)

        qp.end()

    def darw_polygon(self, qp):
        points1 = self.polygon_points
        polygon1 = QPolygon(points1)

        # line
        qp.setPen(QPen(QColor(96, 96, 255), 3))

        # color
        qp.setBrush(QBrush(QColor(196, 196, 255, 96)))

        # draw
        qp.drawPolygon(polygon1)
