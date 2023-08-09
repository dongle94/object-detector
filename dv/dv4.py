import os
import sys
import argparse
import cv2
import time
import numpy as np
import shapely
from datetime import datetime, timedelta
from collections import defaultdict

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QDialog
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QLineEdit, QLabel, QPushButton
from PySide6.QtWidgets import QFileDialog
from PySide6.QtCore import QSize, Qt, Slot, QPoint, QThread
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

from utils.medialoader import MediaLoader, check_sources
from gui.image import ImgWidget
from gui.widget import MsgDialog

from core.bbox import BBox
from core.obj_detectors import ObjectDetector
from core.tracking import ObjectTracker


class ResultDialog(QDialog):
    def __init__(self):
        pass
        # 존재하는 분석 결과 파일 읽어와서 보여주기


class AnalysisResult(object):
    def __init__(self, source):
        self.data = {}
        self.source = source


    def init_analysis(self):
        get_logger().info("AnalysisResult initialization.")
        self.data = {
            'source': self.source,
            'start_time': datetime.now(),
            'end_time': None,
            'object_cnt': defaultdict(int),
            'tts': defaultdict(float),
            'checked_id': []
        }

    def stop_analysis(self):
        pass

    def save_result(self):
        file_name = f'{datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}.txt'
        with open(file_name, mode='w', encoding='utf8') as f:
            f.write("hello")


class AnalysisThread(QThread):
    def __init__(self, parent, input_path=None, detector=None, tracker=None, polygons=[]):
        super().__init__(parent=parent)
        self.logger = get_logger()
        self.logger.info("Create AnalysisThread.")

        self.sbar = self.parent().sbar
        self.polygons = polygons

        self.is_file, self.is_url, self.is_webcam = check_sources(input_path)
        self.input_path = input_path
        if self.is_url or self.is_file:
            realtime = True if self.is_url else True   # else is is_file
            self.medialoader = MediaLoader(source=input_path, realtime=realtime)
            self.logger.info(f"Non Webcam Medialoader source: {input_path} / realtime: {realtime}")
            self.medialoader.start()
            self.medialoader.pause()

        # analysis module
        self.detector = detector
        self.tracker = tracker

        self.viewer = self.parent().layer_1
        self.analysis_result = AnalysisResult(input_path)
        self.stop_run = False

        self.id_cnt = defaultdict(int)
        self.t_box_data = {}

        # analysis logging
        self.f_cnt = 0
        self.log_interval = self.parent().config.CONSOLE_LOG_INTERVAL
        self.ts = [0., 0., 0., ]

    def run(self):
        self.logger.info("영상 분석 시작")
        self.sbar.showMessage("영상 분석 시작")
        self.analysis_result.init_analysis()
        if self.is_webcam:
            self.medialoader = MediaLoader(source=self.input_path, realtime=True)
            self.logger.info(f"Webcam Medialoader source: {self.input_path} / realtime: {True}")
            self.medialoader.start()
        else:
            self.medialoader.restart()
        self.stop_run = False
        fps = self.medialoader.fps
        w_time = 1 / fps

        while True:
            frame = self.medialoader.get_frame()
            if frame is None or self.stop_run is True:
                break

            if frame.shape[0] >= 1080:
                frame = cv2.resize(frame, (int(frame.shape[1]*0.8), int(frame.shape[0]*0.8)))

            t0 = time.time()
            im = self.detector.preprocess(frame)
            _pred = self.detector.detect(im)
            _pred, _det = self.detector.postprocess(_pred)

            boxes = []
            # analysis area filtering
            for _d in _det:
                x1, y1, x2, y2 = _d[:4]
                w, h = int(x2-x1), int(y2-y1)
                bx, by = int(x1 + 0.5 * w), int(y1 + 0.9 * h)
                bench_point = shapely.Point(bx, by)
                for polygon in self.polygons:
                    if polygon.contains(bench_point):
                        cv2.line(frame, (bx, by), (bx, by),
                                 (224, 96, 255), 5, cv2.LINE_AA)
                        boxes.append(_d)
                        break
            boxes = np.array(boxes)
            t1 = time.time()

            _boxes = []
            # tracking and creating BBox instance
            if len(boxes):
                track_ret = self.tracker.update(boxes, frame)
                if len(track_ret):
                    t_boxes = track_ret[:, 0:4].astype(np.int32)
                    t_ids = track_ret[:, 4].astype(np.int32)
                    t_confs = track_ret[:, 5]
                    t_classes = track_ret[:, 6]
                    for xyxy, _id, conf, cls in zip(t_boxes, t_ids, t_confs, t_classes):
                        if _id != -1:
                            bbox = BBox(tlbr=xyxy,
                                        class_index=int(cls),
                                        class_name=self.detector.names[int(cls)],
                                        conf=conf,
                                        imgsz=frame.shape)
                            bbox.tracking_id = _id
                            _boxes.append(bbox)

                            # 분석결과 처리
                            if self.id_cnt[_id] == 0:       # 첫 등장
                                self.t_box_data[_id] = {
                                    'class_index': int(cls),
                                    'access_time': time.time()
                                }
                            elif self.id_cnt[_id] == 10:  # 실제 카운트 인정
                                if _id not in self.analysis_result.data['checked_id']:
                                    self.analysis_result.data['object_cnt'][bbox.class_name] += 1
                                    self.analysis_result.data['checked_id'].append(_id)
                                self.t_box_data[_id]['last_modified_time'] = time.time()
                            elif self.id_cnt[_id] > 10:
                                self.t_box_data[_id]['last_modified_time'] = time.time()
                            self.id_cnt[_id] += 1

            del_list = []
            for _id, data in self.t_box_data.items():
                if'last_modified_time' in data:
                    if time.time() - self.t_box_data[_id]['last_modified_time'] > 10:   # 유효 박스 중 사라진 박스
                        _class_idx = self.t_box_data[_id]['class_index']
                        self.analysis_result.data['tts'][_class_idx] += (self.t_box_data[_id]['last_modified_time'] - self.t_box_data[_id]['access_time'])
                        del_list.append(_id)
                        print(_id, self.analysis_result.data)
                else:       # 등장만하고 유효하지 않은 박스
                    if time.time() - self.t_box_data[_id]['access_time'] > 10:
                        del_list.append(_id)
            for _id in del_list:
                del self.id_cnt[_id]
                del self.t_box_data[_id]

            t2 = time.time()

            # visualize
            for b in _boxes:
                x1, y1, x2, y2 = b.x1, b.y1, b.x2, b.y2
                if b.tracking_id != -1:
                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  color=(96, 216, 96),
                                  thickness=2, lineType=cv2.LINE_AA)
                    cv2.putText(frame, text=f"({b.class_name})ID: {b.tracking_id}",
                                org=(b.x1, b.y1+10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                                color=(96, 96, 216), thickness=2)

            self.viewer.img_label.draw.emit(frame, True)
            t3 = time.time()

            # calculate time
            self.ts[0] += (t1 - t0)
            self.ts[1] += (t2 - t1)
            self.ts[2] += (t3 - t2)

            self.f_cnt += 1
            if self.f_cnt % self.log_interval == 0:
                self.logger.debug(
                    f"[{self.f_cnt} Frames] det: {self.ts[0] / self.f_cnt:.4f} / "
                    f"tracking: {self.ts[1] / self.f_cnt:.4f} / "
                    f"visualize: {self.ts[2] / self.f_cnt:.4f}")
            if (t2 - t0) + 0.001 < w_time:
                s_time = w_time - (t2 - t0) - 0.001
                time.sleep(s_time)

        self.sbar.showMessage("영상 분석 중지")
        get_logger().info("Analysis Thraed - 영상 분석 종료")

    def stop_analysis(self):
        self.analysis_result.data['end_time'] = datetime.now()
        for _id, data in self.t_box_data.items():
            if 'last_modified_time' in data:
                _class_idx = self.t_box_data[_id]['class_index']
                self.analysis_result.data['tts'][_class_idx] += (
                            self.t_box_data[_id]['last_modified_time'] - self.t_box_data[_id]['access_time'])

        self.id_cnt = defaultdict(int)
        self.t_box_data = {}
        print(self.analysis_result.data)

    def stop(self):
        self.medialoader.stop()
        del self.medialoader


class SetAreaDialog(QDialog):
    def __init__(self, img, parent=None):
        super().__init__(parent=parent)

        label = QLabel("3개 이상의 점을 찍어 영역을 설정해 주세요.")
        self.frame = img
        if img.shape[0] >= 1080:
            img = cv2.resize(img, (int(img.shape[1] * 0.8), int(img.shape[0] * 0.8)))
        self.img_size = img.shape
        self.img = ImgWidget(parent=self, polygon=True)
        self.img.set_array(img) #, scale=True)
        self.bt = QPushButton("영역 설정 완료")

        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.img)
        layout.addWidget(self.bt)
        self.setLayout(layout)

        self.setWindowTitle("분석 영역 설정")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        self.polygons = []
        self.polygon = []

        # signals & slots
        self.bt.clicked.connect(self.click_setArea)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            point = event.position()
            point = self.img.img_label.mapFromParent(point)
            x = point.x()-self.img.pos().x()
            y = point.y()-self.img.pos().y()
            if 0 < x < self.img_size[1] and 0 < y < self.img_size[0]:
                new_point = QPoint(point.x()-self.img.pos().x(), point.y()-self.img.pos().y())

                self.polygon.append(new_point)
                if len(self.polygon) == 3:
                    self.img.img_label.polygon_points.append(self.polygon)
                elif len(self.polygon) > 3:
                    self.img.img_label.polygon_points[-1] = self.polygon
                self.img.img_label.repaint()

            else:
                get_logger().error("영역 설정 화면에서 잘못된 좌표를 클릭하였습니다.")

    def click_setArea(self):
        # 창은 숨기기
        self.hide()
        # 폴리곤 인스턴스 셋팅
        p_list = [p.toTuple() for p in self.polygon]
        polygon = shapely.Polygon(p_list)
        self.polygons.append(polygon)

        # Main Widget 으로 전달해주기
        main_widget = self.parent()
        main_widget.analysis_area.append(polygon)
        main_widget.set_image_area(self.frame, self.polygons, f_size=self.img_size)
        # 초기화
        self.polygon = []


class MainWidget(QWidget):
    def __init__(self, cfg, parent=None):
        super().__init__(parent=parent)

        # init Widget
        self.config = cfg
        self.logger = get_logger()
        self.logger.info("Create Main Widget - Start")
        self.sbar = self.parent().statusBar()
        self.set_dialog = None

        # 1
        self.layer_0 = QHBoxLayout()
        self.el_source = QLineEdit()
        self.bt_find_source = QPushButton("동영상 파일 찾기")
        self.bt_set_area = QPushButton("분석 영역 설정")
        self.bt_reset = QPushButton("초기화")
        self.bt_set_area.setDisabled(True)

        self.layer_0.addWidget(QLabel("미디어 소스: "))
        self.layer_0.addWidget(self.el_source)
        self.layer_0.addWidget(self.bt_find_source)
        self.layer_0.addWidget(self.bt_set_area)
        self.layer_0.addWidget(self.bt_reset)

        # 2
        self.layer_1 = ImgWidget(parent=self, polygon=True)
        self.layer_1.set_file('./data/images/default-video.png')

        # 3
        self.layer_2 = QHBoxLayout()
        self.bt_start = QPushButton("분석 시작")
        self.bt_stop = QPushButton("분석 종료")
        self.bt_result = QPushButton("분석 결과 보기")

        self.layer_2.addWidget(self.bt_start)
        self.layer_2.addWidget(self.bt_stop)
        self.layer_2.addWidget(self.bt_result)
        self.bt_start.setDisabled(True)
        self.bt_stop.setDisabled(True)
        self.bt_result.setDisabled(True)

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
        self.bt_reset.clicked.connect(self.reset)
        self.bt_start.clicked.connect(self.start_analysis)
        self.bt_stop.clicked.connect(self.stop_analysis)
        self.layer_1.img_label.draw.connect(self.draw_img)

        # Op
        self.analysis_area = []

        self.obj_detector = ObjectDetector(cfg=self.config)
        self.obj_tracker = ObjectTracker(cfg=self.config)
        self.analysis_thread = None

    @Slot()
    def find_video(self):
        f_name = QFileDialog.getOpenFileName(parent=self, caption="비디오 파일 선택",
                                             filter="All Files(*);;"
                                                    "Videos(*.webm);;"
                                                    "Videos(*.mp4 *.avi *m4v *.mpg *mpeg);;"
                                                    "Videos(*.wmv *.mov *.mkv *.flv)")
        self.el_source.setText(f_name[0])
        self.bt_set_area.setDisabled(False)

        self.sbar.showMessage("분석 시작을 위해서 분석 영역 설정을 해 주세요.")

    @Slot()
    def enable_bt_set_area(self):
        if len(self.el_source.text()) == 0:
            self.bt_set_area.setDisabled(True)

            self.bt_start.setDisabled(True)
            self.bt_stop.setDisabled(True)
            self.bt_result.setDisabled(True)
        else:
            self.bt_set_area.setDisabled(False)

            self.sbar.showMessage("분석 시작을 위해서 분석 영역 설정을 해 주세요.")

    @Slot()
    def set_area(self):
        path = self.el_source.text()
        try:
            ml = MediaLoader(path, logger=get_logger(), realtime=False)
            ml.start()
            f = None
            for _ in range(2):
                f = ml.get_frame()
            if self.set_dialog is None:
                self.set_dialog = SetAreaDialog(img=f, parent=self)
            self.set_dialog.show()
            ml.stop()
        except Exception as e:
            self.logger.warning(e)
            MsgDialog(parent=self,
                      msg="Not Exist Input File.\n"
                          "Please Check source path.",
                      title="Error Input")

    def set_image_area(self, img, polygons, f_size):
        # image size check
        img_size = (f_size[1], f_size[0])

        # img label size check
        self.layer_1.set_array(img, scale=True)
        lbl_size = (self.layer_1.img_label.size().width(), self.layer_1.img_label.size().height())

        # adjust polygon points
        for polygon in polygons:
            p_list = []
            for point in polygon.exterior.coords[:-1]:
                orig_x, orig_y = point[0], point[1]
                new_x, new_y = int(orig_x / img_size[0] * lbl_size[0]), int(orig_y / img_size[1] * lbl_size[1])
                new_point = QPoint(new_x, new_y)
                p_list.append(new_point)
            self.layer_1.img_label.polygon_points.append(p_list)

        self.layer_1.resize(img.shape[1], img.shape[0])
        self.layer_1.img_label.repaint()

        # if button state is disabled, enable button when you set area.
        self.bt_start.setDisabled(False)
        self.bt_stop.setDisabled(False)
        self.bt_result.setDisabled(False)

        if self.analysis_thread is None:
            self.analysis_thread = AnalysisThread(
                parent=self,
                input_path=self.el_source.text(),
                detector=self.obj_detector,
                tracker=self.obj_tracker,
                polygons=self.analysis_area
            )

    @Slot()
    def reset(self):
        self.analysis_area = []
        self.layer_1.img_label.polygon_points = []
        self.set_dialog = None

        if self.analysis_thread is not None:
            self.analysis_thread.stop_run = True
            time.sleep(0.1)
            self.analysis_thread.stop()
        self.analysis_thread = None

        self.bt_start.setDisabled(True)
        self.bt_stop.setDisabled(True)
        self.bt_result.setDisabled(True)

        self.layer_1.set_file('./data/images/default-video.png')
        self.logger.info("Disable button / img dialog and analysis thread be empty")

    @Slot()
    def start_analysis(self):
        self.logger.info("Start analysis")

        self.analysis_thread.start()

    @Slot()
    def stop_analysis(self):
        self.logger.info("Stop analysis")

        self.analysis_thread.stop_run = True
        self.analysis_thread.exit()
        self.analysis_thread.stop_analysis()

    @Slot()
    def draw_img(self, img, scale=False):
        self.layer_1.set_array(img, scale)


class WithYou(QMainWindow):
    def __init__(self, config=None):
        super().__init__()

        # init gui
        self.logger = get_logger()
        self.logger.info("Init WithYou QMainWindow - start")

        # resize window
        self.geo = self.screen().availableGeometry()
        self.setFixedSize(self.geo.width() * 0.8, self.geo.height() * 0.8)

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
