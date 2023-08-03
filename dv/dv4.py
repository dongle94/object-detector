import os
import sys
import argparse
import cv2
import numpy as np

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QDialog
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QLineEdit, QLabel, QPushButton
from PySide6.QtCore import QSize, Qt

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
os.chdir(ROOT)

from utils.config import _C as cfg
from utils.config import update_config
from utils.logger import init_logger, get_logger

from gui.image import ImgWidget, ImgDialog


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
