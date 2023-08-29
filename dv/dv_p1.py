# -*- coding: utf-8 -*-
"""
datavoucher P1 - 와이 이노베이션
Industrial Garbage detection demo
"""

import os
import sys
import argparse
import cv2

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PySide6.QtCore import Qt, QSize, Slot, QThread, QRect

from utils.config import _C as cfg, update_config
from utils.logger import init_logger, get_logger
from gui.image import ImgWidget, EllipseLabel


class MainWidget(QWidget):
    def __init__(self, cfg, parent=None):
        super().__init__(parent=parent)

        # init MainWidget
        self.config = cfg
        self.logger = get_logger()
        self.logger.info("Create Main QWidget - start")

        self.setupUi()

    def setupUi(self):
        self.frame_width = self.parent().size().width()
        self.frame_height = self.parent().size().height()

        # layout 0
        self.layer_0 = QHBoxLayout()
        self.layer_0_left = QVBoxLayout()
        self.img_view = ImgWidget(parent=self, polygon=False)
        self.img_view.set_file('./data/images/default-video1.png')
        self.img_view.setMinimumWidth(int(self.frame_width * 0.7))

        self.layer_0_right = QVBoxLayout()
        self.lb_help_status = QHBoxLayout()
        txt_1 = QLabel("영상 분석 상태: ")
        # green (0,150,75) / yellow (255,216,32) / red (216,3232)
        draw_1 = EllipseLabel(size=(3, 3, 30, 20),
                              line_color=(0, 0, 0),
                              fill_color=(0, 150, 75))

        self.lb_result = QLabel("분석결과")
        self.lb_result.setMinimumHeight(int(self.frame_height * 0.9))
        self.lb_result.setStyleSheet("border: 1px solid black;")

        self.layer_0_left.addWidget(self.img_view)
        self.layer_0.addLayout(self.layer_0_left)

        self.lb_help_status.addWidget(txt_1, 10)
        self.lb_help_status.addWidget(draw_1, 90)

        self.layer_0_right.addLayout(self.lb_help_status)
        self.layer_0_right.addWidget(self.lb_result)
        self.layer_0.addLayout(self.layer_0_right)

        # layout 1
        self.layer_1 = QHBoxLayout()
        self.bt_find_video = QPushButton("동영상 파일 찾기")
        self.bt_start = QPushButton("분석 시작")
        self.bt_stop = QPushButton("분석 종료")

        self.layer_1.addWidget(self.bt_find_video)
        self.layer_1.addWidget(self.bt_start)
        self.layer_1.addWidget(self.bt_stop)

        # Main Widget
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        self.main_layout.addLayout(self.layer_0)
        self.main_layout.addLayout(self.layer_1)

class P1(QMainWindow):
    def __init__(self, config=None):
        super().__init__()

        # init gui
        self.logger = get_logger()
        self.logger.info("Init P1 QMainWindow - start")

        # resize window
        self.geo = self.screen().availableGeometry()
        self.setFixedSize(self.geo.width() * 0.8, self.geo.height() * 0.8)

        # Set Central Widget
        self.main_widget = MainWidget(cfg=config, parent=self)
        self.setCentralWidget(self.main_widget)

        self.logger.info("Init P1 QMainWindow - end")


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./configs/dvp1.yaml',
                        help='configuration')
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

    app_gui = P1(cfg)
    app_gui.setWindowTitle("P1")
    app_gui.show()

    sys.exit(app.exec())
