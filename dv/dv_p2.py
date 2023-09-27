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
import numpy as np
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
from gui.widget import NormalLabel

from utils.config import _C as cfg, update_config
from utils.logger import init_logger, get_logger
from utils.medialoader import MediaLoader
from core.obj_detectors import ObjectDetector
from core.tracking import ObjectTracker


def mosaic(img: np.ndarray, coord, block=10):
    x1, y1, x2, y2 = coord
    w, h = x2 - x1+1, y2 - y1+1
    if w < block:
        block /= 2
    gap_w, gap_h = int(w/block), int(h/block)
    if gap_w == 0 or gap_h == 0:
        return
    for c in range(3):
        for ny in range(y1, y2, gap_h):
            for nx in range(x1, x2, gap_w):
                new_y = ny + gap_h + 1 if ny + gap_h < y2 else y2
                new_x = nx + gap_w + 1 if nx + gap_w < x2 else x2
                img[ny:new_y, nx:new_x, c] = np.mean(img[ny:new_y, nx:new_x, c])


class AnalysisThread(QThread):
    def __init__(self, parent, input_path=None, detector=None, head_detector=None, tracker=None, img_viewer=None):
        super().__init__(parent=parent)
        self.logger = get_logger()
        self.cfg = parent.config

        # input source
        self.input_path = input_path
        self.media_loader = MediaLoader(source=input_path, logger=self.logger, realtime=False, fast=True)
        self.viewer = img_viewer
        self.side_viewers = [self.parent().imgWidget_0, self.parent().imgWidget_1, self.parent().imgWidget_2,
                             self.parent().imgWidget_3, self.parent().imgWidget_4, self.parent().imgWidget_5]
        self.lb_result = self.parent().num_process
        self.num_frames = int(self.media_loader.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # output source
        self.vw = None

        # draw widget
        self.ellipse = self.parent().draw_1
        self.pbar = self.parent().pbar

        # analysis module
        self.detector = detector
        self.head_detector = head_detector
        self.tracker = tracker

        # analysis logging
        self.id_cnt = defaultdict(int)
        self.class_cnt = defaultdict(int)
        for v in self.detector.names.values():
            self.class_cnt[v] = 0

        self.f_cnt = 0
        self.log_interval = self.parent().config.CONSOLE_LOG_INTERVAL
        self.ts = [0., 0., 0., ]

    def run(self) -> None:
        self.logger.info("Start analysis.")

        self.media_loader.start()
        self.ellipse.updateFillColor.emit((216, 32, 32))
        while True:
            frame = self.media_loader.get_frame()
            if frame is None or len(frame.shape) < 2:
                break

            # Detection
            t0 = time.time()

            im = self.detector.preprocess(frame)
            _pred = self.detector.detect(im)
            _pred0, _det0 = self.detector.postprocess(_pred)

            im = self.head_detector.preprocess(frame)
            _pred = self.head_detector.detect(im)
            _pred1, _det1 = self.head_detector.postprocess(_pred)

            t1 = time.time()

            # Tracking
            _boxes = []
            if len(_det0):
                track_ret = self.tracker.update(_det0, frame)
                if len(track_ret):
                    for _t in track_ret:
                        xyxy = _t[:4]
                        _id = int(_t[4])
                        conf = _t[5]
                        cls = _t[6]
            t2 = time.time()

            for _box in _boxes:
                self.id_cnt[_box.tracking_id] += 1      # register count

                if self.id_cnt[_box.tracking_id] == self.cfg.REGISTER_FRAME:      # real object count
                    self.class_cnt[_box.class_name] += 1

                    # update
                    _txt = ''
                    for k, v in self.class_cnt.items():
                        if v != 'background':
                            _txt += f"{k:10s} : {v}\n"
                    self.lb_result.updateText.emit(_txt)

            for d in _det0:
                x1, y1, x2, y2 = map(int, d[:4])
                mosaic(frame, (x1, y1, x2, y2), block=10)
            for d in _det1:
                if d[5] == 1:
                    x1, y1, x2, y2 = map(int, d[:4])
                    mosaic(frame, (x1, y1, x2, y2), block=10)

            self.viewer.img_label.draw.emit(frame, True)

            self.vw.write(frame)

            t3 = time.time()

            # calculate time
            self.ts[0] += (t1 - t0)
            self.ts[1] += (t2 - t1)
            self.ts[2] += (t3 - t2)

            # logging
            self.f_cnt += 1
            if self.f_cnt % self.log_interval == 0:
                self.logger.info(
                    f"[{self.f_cnt} Frame] det: {self.ts[0] / self.f_cnt:.4f} / "
                    f"tracking: {self.ts[1] / self.f_cnt:.4f} / "
                    f"visualize and writing: {self.ts[2] / self.f_cnt:.4f}"
                )

            # Process Event
            if int(self.num_frames * 1 / 7) == self.f_cnt:
                self.side_viewers[0].img_label.draw.emit(frame, True)
            elif int(self.num_frames * 2 / 7) == self.f_cnt:
                self.side_viewers[1].img_label.draw.emit(frame, True)
            elif int(self.num_frames * 3 / 7) == self.f_cnt:
                self.side_viewers[2].img_label.draw.emit(frame, True)
            elif int(self.num_frames * 4 / 7) == self.f_cnt:
                self.side_viewers[3].img_label.draw.emit(frame, True)
            elif int(self.num_frames * 5 / 7) == self.f_cnt:
                self.side_viewers[4].img_label.draw.emit(frame, True)
            elif int(self.num_frames * 6 / 7) == self.f_cnt:
                self.side_viewers[5].img_label.draw.emit(frame, True)

            self.pbar.valueChanged.emit(int(self.f_cnt / self.num_frames * 100))

        # End Process
        self.ellipse.updateFillColor.emit((0, 150, 75))
        self.vw.release()

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
        self.c_imgWidget.img_label.draw.connect(self.draw_img)
        self.imgWidget_0.img_label.draw.connect(self.draw_img0)
        self.imgWidget_1.img_label.draw.connect(self.draw_img1)
        self.imgWidget_2.img_label.draw.connect(self.draw_img2)
        self.imgWidget_3.img_label.draw.connect(self.draw_img3)
        self.imgWidget_4.img_label.draw.connect(self.draw_img4)
        self.imgWidget_5.img_label.draw.connect(self.draw_img5)
        self.draw_1.updateFillColor.connect(self.change_fill_color)
        self.pbar.valueChanged.connect(self.pbar.setValue)
        self.num_process.updateText.connect(self.update_text)

        # Analysis
        self.obj_detector = ObjectDetector(cfg=self.config)
        self.config.defrost()
        self.config.DET_MODEL_PATH = './weights/crowdhuman_yolov5m.pt'
        self.config.IMG_SIZE = 640
        self.config.freeze()
        self.head_detector = ObjectDetector(cfg=self.config)
        self.obj_tracker = ObjectTracker(cfg=self.config)
        self.analysis_thread = None

        # post update
        txt = ''
        for v in self.obj_detector.names.values():
            if v != 'background':
                txt += f"{v:10s} : 0\n"
        self.num_process.setText(txt)

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
        self.imgWidget_0.setFixedSize(int(self.frame_width * 0.3), int(self.frame_height * 0.3))
        self.imgWidget_1 = ImgWidget()
        self.imgWidget_1.setFixedSize(int(self.frame_width * 0.3), int(self.frame_height * 0.3))
        self.imgWidget_2 = ImgWidget()
        self.imgWidget_2.setFixedSize(int(self.frame_width * 0.3), int(self.frame_height * 0.3))
        self.imgWidget_3 = ImgWidget()
        self.imgWidget_3.setFixedSize(int(self.frame_width * 0.3), int(self.frame_height * 0.3))
        self.imgWidget_4 = ImgWidget()
        self.imgWidget_4.setFixedSize(int(self.frame_width * 0.3), int(self.frame_height * 0.3))
        self.imgWidget_5 = ImgWidget()
        self.imgWidget_5.setFixedSize(int(self.frame_width * 0.3), int(self.frame_height * 0.3))

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
        self.num_process = NormalLabel("")
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
        f_name, _ = QFileDialog.getOpenFileName(parent=self, caption="비디오 파일 선택",
                                             filter="All Files(*);;"
                                                    "Videos(*.webm);;"
                                                    "Videos(*.mp4 *.avi *m4v *.mpg *.mpeg);;"
                                                    "Videos(*.wmv *.mov *.mkv *.flv)")
        self.logger.info(f"Find Video - {f_name}")
        if f_name != "":
            if self.analysis_thread is not None:
                self.analysis_thread = None
                self.c_imgWidget.img_label.clear()
                self.imgWidget_0.img_label.clear()
                self.imgWidget_1.img_label.clear()
                self.imgWidget_2.img_label.clear()
                self.imgWidget_3.img_label.clear()
                self.imgWidget_4.img_label.clear()
                self.imgWidget_5.img_label.clear()
                self.change_fill_color((255, 216, 32))
            if self.analysis_thread is None:
                ml = MediaLoader(f_name, logger=get_logger(), realtime=False)
                f = None
                for _ in range(2):
                    f = ml.get_one_frame()
                del ml
                self.c_imgWidget.set_array(f, True)
                self.analysis_thread = AnalysisThread(
                    parent=self,
                    input_path=f_name,
                    detector=self.obj_detector,
                    head_detector=self.head_detector,
                    tracker=self.obj_tracker,
                    img_viewer=self.c_imgWidget
                )

            self.parent().statusBar().showMessage(f"{f_name}")

    @Slot()
    def process(self):
        self.logger.info("Click - bt_process button.")

        f_name, _ = QFileDialog.getSaveFileName(
            parent=self, caption="비디오 파일 저장",
            dir=os.getenv("HOME"),
            filter="Videos(*.mp4);;")
        if f_name != "":
            filename = f_name + '.mp4' if os.path.splitext(f_name)[1] not in ['.mp4', '.webm', '.avi'] else f_name
            self.logger.info(f"Save Video - {filename}")
            self.analysis_thread.vw = cv2.VideoWriter(
                filename=filename,
                fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                fps=self.analysis_thread.media_loader.fps,
                frameSize=(self.analysis_thread.media_loader.w, self.analysis_thread.media_loader.h),
                isColor=True
            )

            self.analysis_thread.start()

    @Slot()
    def draw_img(self, img, scale=False):
        self.c_imgWidget.set_array(img, scale)

    @Slot()
    def draw_img0(self, img, scale=False):
        self.imgWidget_0.set_array(img, scale)

    @Slot()
    def draw_img1(self, img, scale=False):
        self.imgWidget_1.set_array(img, scale)

    @Slot()
    def draw_img2(self, img, scale=False):
        self.imgWidget_2.set_array(img, scale)

    @Slot()
    def draw_img3(self, img, scale=False):
        self.imgWidget_3.set_array(img, scale)

    @Slot()
    def draw_img4(self, img, scale=False):
        self.imgWidget_4.set_array(img, scale)

    @Slot()
    def draw_img5(self, img, scale=False):
        self.imgWidget_5.set_array(img, scale)

    @Slot()
    def change_fill_color(self, color):
        self.draw_1.fill_color = color
        self.draw_1.update()

    @Slot()
    def update_text(self, txt):
        self.num_process.setText(txt)


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

