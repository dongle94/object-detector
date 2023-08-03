import os
import sys
import argparse
import cv2
import numpy as np

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QDialog
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QLineEdit, QLabel, QPushButton
from PySide6.QtWidgets import QFileDialog
from PySide6.QtCore import QSize, Qt, Slot, QPoint
from PySide6.QtGui import QMouseEvent

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
os.chdir(ROOT)

from utils.config import _C as cfg
from utils.config import update_config
from utils.logger import init_logger, get_logger

from utils.medialoader import MediaLoader
from gui.image import ImgWidget, ImgDialog
from gui.widget import MsgDialog


class SetAreaDialog(QDialog):
    def __init__(self, img, parent=None):
        super().__init__(parent=parent)

        label = QLabel("3개 이상의 점을 찍어 영역을 설정해 주세요.")
        self.img_size = img.shape
        self.img = ImgWidget(parent=self, polygon=True)
        self.img.set_array(img)
        self.bt = QPushButton("영역 설정 완료")

        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.img)
        layout.addWidget(self.bt)
        self.setLayout(layout)

        self.setWindowTitle("분석 영역 설정")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            point = event.position()
            point = self.img.img_label.mapFromParent(point)
            x = point.x()-self.img.pos().x()
            y = point.y()-self.img.pos().y()
            if 0 < x < self.img_size[1] and 0 < y < self.img_size[0]:
                new_point = QPoint(point.x()-self.img.pos().x(), point.y()-self.img.pos().y())
                self.img.img_label.polygon_points.append(new_point)
                if len(self.img.img_label.polygon_points) >= 3:
                    self.img.img_label.repaint()
            else:
                get_logger().error("영역 설정 화면에서 잘못된 좌표를 클릭하였습니다.")


class MainWidget(QWidget):
    def __init__(self, cfg, parent=None):
        super().__init__(parent=parent)

        # init Widget
        self.config = cfg
        self.logger = get_logger()
        self.logger.info("Create Main Widget - Start")
        self.sbar = self.parent().statusBar()

        # 1
        self.layer_0 = QHBoxLayout()
        self.el_source = QLineEdit()
        self.bt_find_source = QPushButton("동영상 파일 찾기")
        self.bt_set_area = QPushButton("분석 영역 설정")
        self.bt_set_area.setDisabled(True)

        self.layer_0.addWidget(QLabel("미디어 소스: "))
        self.layer_0.addWidget(self.el_source)
        self.layer_0.addWidget(self.bt_find_source)
        self.layer_0.addWidget(self.bt_set_area)

        # 2
        self.layer_1 = ImgWidget(parent=self)
        self.layer_1.set_file('./data/images/default-video.png')

        # 3
        self.layer_2 = QHBoxLayout()
        self.bt_start = QPushButton("분석 시작")
        self.bt_stop = QPushButton("분석 중지")
        self.bt_result = QPushButton("분석 결과 보기")

        self.layer_2.addWidget(self.bt_start)
        self.layer_2.addWidget(self.bt_stop)
        self.layer_2.addWidget(self.bt_result)

        # main layout
        self.main = QVBoxLayout()
        self.setLayout(self.main)
        self.main.addLayout(self.layer_0)
        self.main.addWidget(self.layer_1)
        self.main.addLayout(self.layer_2)

        # signals & slots
        self.bt_find_source.clicked.connect(self.find_video)
        self.el_source.textEdited.connect(self.enable_bt_set_area)
        self.bt_set_area.clicked.connect(self.set_area)

    @Slot()
    def find_video(self):
        f_name = QFileDialog.getOpenFileName(parent=self, caption="비디오 파일 선택",
                                             filter="All Files(*);;"
                                                    "Videos(*.webm);;"
                                                    "Videos(*.mp4 *.avi *m4v *.mpg *mpeg);;"
                                                    "Videos(*.wmv *.mov *.mkv *.flv)")
        self.el_source.setText(f_name[0])
        self.bt_set_area.setDisabled(False)

    @Slot()
    def enable_bt_set_area(self):
        if len(self.el_source.text()) == 0:
            self.bt_set_area.setDisabled(True)
        else:
            self.bt_set_area.setDisabled(False)

    @Slot()
    def set_area(self):
        path = self.el_source.text()
        try:
            ml = MediaLoader(path, logger=get_logger(), realtime=False)
            ml.start()
            f = None
            for _ in range(2):
                f = ml.get_frame()
            set_dialog = SetAreaDialog(img=f, parent=self)
            set_dialog.show()
        except Exception as e:
            self.logger.warning(e)
            MsgDialog(parent=self,
                      msg="Not Exist Input File.\n"
                          "Please Check source path.",
                      title="Error Input")


class WithYou(QMainWindow):
    def __init__(self, config=None):
        super().__init__()

        # init gui
        self.logger = get_logger()
        self.logger.info("Init WithYou QMainWindow - start")

        # resize window
        self.geo = self.screen().availableGeometry()
        self.resize(QSize(self.geo.width() * 0.7, self.geo.height() * 0.7))

        # Status Bar
        self.sbar = self.statusBar()
        self.sbar.showMessage("프로그램 준비", 0)

        # Set Widget
        self.main_widget = MainWidget(cfg=cfg, parent=self)
        self.setCentralWidget(self.main_widget)

        self.logger.info("Init WithYou QMainWindow - end")


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='configuration', default='./configs/dv4.yaml')
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Get Configuration
    args = args_parse()
    _cfg = args.config
    update_config(cfg, _cfg)

    # initialize logger
    init_logger(cfg=cfg)

    app_gui = WithYou(cfg)
    app_gui.setWindowTitle("위드유컴퍼니")
    app_gui.show()

    sys.exit(app.exec())
