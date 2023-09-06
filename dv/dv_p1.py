# -*- coding: utf-8 -*-
"""
datavoucher P1 - 와이 이노베이션
Industrial Garbage detection demo
"""

import os
import sys
import argparse
import time
import cv2
from collections import defaultdict

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog
from PySide6.QtCore import Qt, Slot, QThread, Signal
from PySide6.QtGui import QFont

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
os.chdir(ROOT)

from gui.image import ImgWidget, EllipseLabel
from gui.widget import NormalLabel

from core.bbox import BBox
from utils.medialoader import MediaLoader
from utils.config import _C as cfg, update_config
from utils.logger import init_logger, get_logger
from core.obj_detectors import ObjectDetector
from core.tracking import ObjectTracker


class AnalysisThread(QThread):
    def __init__(self, parent, input_path=None, detector=None, tracker=None, img_viewer=None):
        super().__init__(parent=parent)
        self.logger = get_logger()
        self.cfg = parent.config

        # input source
        self.input_path = input_path
        self.media_loader = MediaLoader(source=input_path, logger=self.logger, realtime=False)
        self.media_loader.start()
        self.media_loader.pause()
        self.viewer = img_viewer
        self.lb_result = self.parent().lb_result

        # analysis module
        self.detector = detector
        self.tracker = tracker

        self.stop_run = False

        # analysis logging
        self.class_cnt = defaultdict(int)
        self.id_cnt = defaultdict(int)

        self.f_cnt = 0
        self.log_interval = self.parent().config.CONSOLE_LOG_INTERVAL
        self.ts = [0., 0., 0., ]

        # show result
        for v in self.detector.names.values():
            self.class_cnt[v] = 0
        self.ellipse = self.parent().draw_1

    def run(self) -> None:
        self.logger.info("Start analysis.")

        self.media_loader.restart()
        self.stop_run = False
        fps = self.media_loader.fps
        w_time = 1 / fps

        while True:
            frame = self.media_loader.get_frame()
            if frame is None or self.stop_run is True:
                break

            if frame.shape[0] >= 1080:
                frame = cv2.resize(frame, (int(frame.shape[1]*0.8), int(frame.shape[0]*0.8)))

            # Detection
            t0 = time.time()

            im = self.detector.preprocess(frame)
            _pred = self.detector.detect(im)
            _pred, _det = self.detector.postprocess(_pred)

            t1 = time.time()

            # Tracking
            _boxes = []
            if len(_det):
                self.ellipse.updateFillColor.emit((216, 32, 32))
                track_ret = self.tracker.update(_det, frame)
                if len(track_ret):
                    for _t in track_ret:
                        xyxy = _t[:4]
                        _id = int(_t[4])
                        conf = _t[5]
                        cls = _t[6]

                        if _id != -1:
                            bbox = BBox(tlbr=xyxy,
                                        class_index=int(cls),
                                        class_name=self.detector.names[int(cls)],
                                        conf=conf,
                                        imgsz=frame.shape)
                            bbox.tracking_id = _id
                            _boxes.append(bbox)
            else:
                self.ellipse.updateFillColor.emit((0, 150, 75))
            t2 = time.time()

            for _box in _boxes:
                self.id_cnt[_box.tracking_id] += 1          # 등록 횟수

                if self.id_cnt[_box.tracking_id] == self.cfg.REGISTER_FRAME:      # 실제 카운트 인정하기 위한 등록 횟수
                    self.class_cnt[_box.class_name] += 1
                    # 분석 결과 업데이트

                    _txt = ''
                    for k, v in self.class_cnt.items():
                        _txt += f"{k:10s} : {v}\n"
                    self.lb_result.updateText.emit(_txt)

            # visualize
            for b in _boxes:
                cv2.rectangle(frame, (b.x1, b.y1), (b.x2, b.y2), (48, 48, 216), thickness=2, lineType=cv2.LINE_AA)

            self.viewer.img_label.draw.emit(frame, True)
            t3 = time.time()

            # calculate time
            self.ts[0] += (t1 - t0)
            self.ts[1] += (t2 - t1)
            self.ts[2] += (t3 - t2)

            # logging
            self.f_cnt += 1
            if self.f_cnt % self.log_interval == 0:
                self.logger.debug(
                    f"[{self.f_cnt} Frame] det: {self.ts[0] / self.f_cnt:.4f} / "
                    f"tracking: {self.ts[1] / self.f_cnt:.4f} / "
                    f"visualize: {self.ts[2] / self.f_cnt:.4f}")

            if (t2 - t0) + 0.001 < w_time:
                s_time = w_time - (t2 - t0) - 0.001
                time.sleep(s_time)

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
        self.bt_start.clicked.connect(self.start_analysis)
        self.bt_stop.clicked.connect(self.stop_analysis)
        self.img_view.img_label.draw.connect(self.draw_img)
        self.draw_1.updateFillColor.connect(self.change_fill_color)
        self.lb_result.updateText.connect(self.update_text)

        # Analysis
        self.obj_detector = ObjectDetector(cfg=self.config)
        self.obj_tracker = ObjectTracker(cfg=self.config)
        self.analysis_thread = None

        # post update
        txt = ''
        for v in self.obj_detector.names.values():
            txt += f"{v:10s} : 0\n"
        self.lb_result.setText(txt)

    def setupUi(self):
        self.frame_width = self.parent().size().width()
        self.frame_height = self.parent().size().height()

        # layout 0
        self.layer_0 = QHBoxLayout()
        self.layer_0_left = QVBoxLayout()
        self.img_view = ImgWidget(parent=self, polygon=False)
        self.img_view.set_file('./data/images/default-video1.png')
        self.img_view.setMinimumWidth(int(self.frame_width * 0.7))
        self.img_view.setMaximumWidth(int(self.frame_width * 0.7))

        self.layer_0_right = QVBoxLayout()
        self.lb_help_status = QHBoxLayout()
        txt_1 = QLabel("영상 분석 상태: ")
        # green (0,150,75) / yellow (255,216,32) / red (216,32,32)
        self.draw_1 = EllipseLabel(size=(3, 3, 30, 20),
                              line_color=(0, 0, 0),
                              fill_color=(255, 216, 32))

        self.lb_result = NormalLabel("분석결과")
        self.lb_result.setFont(QFont('Consolas', 13))
        self.lb_result.setMinimumHeight(int(self.frame_height * 0.88))

        self.layer_0_left.addWidget(self.img_view)
        self.layer_0.addLayout(self.layer_0_left)

        self.lb_help_status.addWidget(txt_1, 10)
        self.lb_help_status.addWidget(self.draw_1, 90)

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
                tracker=self.obj_tracker,
                img_viewer=self.img_view
            )

        self.parent().statusBar().showMessage(f"{f_name[0]}")

    @Slot()
    def start_analysis(self):
        self.logger.info("Click - bt_start button.")

        self.analysis_thread.start()

    @Slot()
    def stop_analysis(self):
        self.logger.info("Click - bt_stop button.")

        self.analysis_thread.stop_run = True
        self.analysis_thread.exit()

    @Slot()
    def draw_img(self, img, scale=False):
        self.img_view.set_array(img, scale)

    @Slot()
    def update_text(self, txt):
        self.lb_result.setText(txt)

    @Slot()
    def change_fill_color(self, color):
        self.draw_1.fill_color = color
        self.draw_1.update()


class P1(QMainWindow):
    def __init__(self, config=None):
        super().__init__()

        # init gui
        self.logger = get_logger()
        self.logger.info("Init P1 QMainWindow - start")

        # resize window
        self.geo = self.screen().availableGeometry()
        self.setFixedSize(self.geo.width() * 0.8, self.geo.height() * 0.8)

        # Status Bar
        self.statusBar().showMessage(" ")

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
