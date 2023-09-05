# -*- coding: utf-8 -*-
"""
datavoucher P2 - 와이 이노베이션
weapon and head mosaic video processing program
"""

import os
import sys
import argparse
import time
import cv2
from collections import defaultdict

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QProgressBar
from PySide6.QtCore import Qt, Slot, Signal, QThread

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
os.chdir(ROOT)

from gui.image import ImgWidget, EllipseLabel

from utils.config import _C as cfg, update_config
from utils.logger import init_logger, get_logger
from utils.medialoader import MediaLoader
from core.obj_detectors import ObjectDetector


class AnalysisThread(QThread):
    def __init__(self, parent, input_path=None, detector=None, head_detector=None, img_viewer=None):
        super().__init__(parent=parent)
        self.logger = get_logger()
        self.cfg = parent.config

        # input source
        self.input_path = input_path
        self.media_loader = MediaLoader(source=input_path, logger=self.logger, realtime=False)
        self.viewer = img_viewer
        self.lb_result = self.parent().num_process

        # analysis module
        self.detector = detector
        self.head_detector = head_detector

        self.class_cnt = defaultdict(int)
        for v in self.detector.names.values():
            self.class_cnt[v] = 0

    def run(self) -> None:
        self.logger.info("Start analysis.")

        self.media_loader.start()
        while True:
            frame = self.media_loader.get_frame()
            if frame is None:
                break

            self.viewer.img_label.draw.emit(frame, True)


        self.logger.info("Analysis Thraed - 영상 분석 종료")


class MainWidget(QWidget):
    def __init__(self, cfg, parent=None):
        super().__init__(parent=parent)

        # init MainWidget
        self.config = cfg
        self.logger = get_logger()
        self.logger.info("Create Main QWidget - start")

        self.setupUi()

        # Slots & Signals
        self.bt_find_video.clicked.connect(self.find_video)
        self.bt_process.clicked.connect(self.process)

        # Analysis
        self.obj_detector = None
        # self.obj_detector = ObjectDetector(cfg=self.config)
        self.config.defrost()
        self.config.DET_MODEL_PATH = './weights/crowdhuman_yolov5m.pt'
        self.config.freeze()
        self.head_detector = None
        # self.head_detector = ObjectDetector(cfg=self.config)
        self.analysis_thread = None

    def setupUi(self):
        self.frame_width = self.parent().size().width()
        self.frame_height = self.parent().size().height()

        # layout 0
        self.widget_0 = QLabel()
        self.layer_0 = QHBoxLayout()
        self.layer_0_left = QVBoxLayout()
        self.layer_0_middle = QVBoxLayout()
        self.layer_0_right = QVBoxLayout()
        self.imgWidget_0 = ImgWidget()
        self.imgWidget_1 = ImgWidget()
        self.imgWidget_2 = ImgWidget()
        self.imgWidget_3 = ImgWidget()
        self.imgWidget_4 = ImgWidget()
        self.imgWidget_5 = ImgWidget()

        self.layer_0_left.addWidget(self.imgWidget_0)
        self.layer_0_left.addWidget(self.imgWidget_1)
        self.layer_0_left.addWidget(self.imgWidget_2)
        self.layer_0_right.addWidget(self.imgWidget_3)
        self.layer_0_right.addWidget(self.imgWidget_4)
        self.layer_0_right.addWidget(self.imgWidget_5)

        self.c_status = QHBoxLayout()
        # green (0,150,75) / yellow (255,216,32) / red (216,32,32)
        self.draw_1 = EllipseLabel(size=(1, 1, 30, 20), line_color=(0, 0, 0), fill_color=(255, 216, 32))
        self.draw_1.setMinimumHeight(22)
        self.c_status.addWidget(QLabel("영상 분석 상태: "), 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.c_status.addWidget(self.draw_1, 0, Qt.AlignmentFlag.AlignVCenter)

        self.c_mosaic = QHBoxLayout()
        self.num_mosaic = QLabel("0")
        self.c_mosaic.addWidget(QLabel("비 식별처리 건 수 총: "), 45, Qt.AlignmentFlag.AlignRight)
        self.c_mosaic.addWidget(self.num_mosaic, 10, Qt.AlignmentFlag.AlignCenter)
        self.c_mosaic.addWidget(QLabel("건"), 45, Qt.AlignmentFlag.AlignLeft)

        self.c_imgWidget = ImgWidget()
        self.c_imgWidget.setFixedSize(int(self.frame_width * 0.25), int(self.frame_height * 0.25))

        self.pbar = QProgressBar()
        self.pbar.setFixedSize(int(self.frame_width * 0.25), int(self.frame_height * 0.02))

        self.c_result = QVBoxLayout()
        self.num_process = QLabel("")
        self.c_result.addWidget(QLabel("[상세 내용]"), 10, alignment=Qt.AlignmentFlag.AlignCenter)
        self.c_result.addWidget(self.num_process, 90, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.layer_0_middle.addLayout(self.c_status, 7)
        self.layer_0_middle.addLayout(self.c_mosaic, 7)
        self.layer_0_middle.addWidget(self.c_imgWidget, 30, Qt.AlignmentFlag.AlignCenter)
        self.layer_0_middle.addWidget(self.pbar, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.layer_0_middle.addLayout(self.c_result, 55)

        self.widget_0.setLayout(self.layer_0)
        self.layer_0.addLayout(self.layer_0_left, 30)
        self.layer_0.addLayout(self.layer_0_middle, 40)
        self.layer_0.addLayout(self.layer_0_right, 30)

        self.widget_0.setStyleSheet("border: 1px solid black;")
        # self.widget_0.setStyleSheet("background-image: ./data/images/army.png")

        # layout 1
        self.layer_1 = QHBoxLayout()
        self.bt_find_video = QPushButton("동영상 파일 찾기")
        self.bt_process = QPushButton("영상 처리 및 저장")

        self.layer_1.addWidget(self.bt_find_video, 20, Qt.AlignmentFlag.AlignRight)
        self.layer_1.addWidget(self.bt_process, 20, Qt.AlignmentFlag.AlignLeft)

        # Main Widget
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        # self.main_layout.addLayout(self.layer_0, 80)
        self.main_layout.addWidget(self.widget_0, 80)
        self.main_layout.addLayout(self.layer_1, 20)

    @Slot()
    def find_video(self):
        f_name = QFileDialog.getOpenFileName(parent=self, caption="비디오 파일 선택",
                                             filter="All Files(*);;"
                                                    "Videos(*.webm);;"
                                                    "Videos(*.mp4 *.avi *m4v *.mpg *mpeg);;"
                                                    "Videos(*.wmv *.mov *.mkv *.flv)")
        self.logger.info(f"Find Video - {f_name}")

        if self.analysis_thread is None:
            self.analysis_thread = AnalysisThread(
                parent=self,
                input_path=f_name[0],
                detector=self.obj_detector,
                head_detector=self.head_detector,
                img_viewer=self.c_imgWidget
            )

        self.parent().statusBar().showMessage(f"{f_name[0]}")

    @Slot()
    def process(self):
        self.logger.info("Click - bt_process button.")

        self.analysis_thread.start()


class P2(QMainWindow):
    def __init__(self, config=None):
        super().__init__()

        # init gui
        self.logger = get_logger()
        self.logger.info("Init P2 QMainWindow - start")

        # resize window
        self.geo = self.screen().availableGeometry()
        self.setFixedSize(self.geo.width() * 0.6, self.geo.height() * 0.6)

        # Status Bar
        self.statusBar().showMessage(" ")

        # Set Central Widget
        self.main_widget = MainWidget(cfg=config, parent=self)
        self.setCentralWidget(self.main_widget)

        self.logger.info("Init P2 QMainWindow - end")


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./configs/dvp2.yaml',
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

    app_gui = P2(cfg)
    app_gui.setWindowTitle("P2")
    app_gui.show()

    sys.exit(app.exec())

