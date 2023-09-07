import os
import sys
import argparse
import time
import cv2
import numpy as np
from collections import OrderedDict, defaultdict
from PIL import ImageFont, ImageDraw, Image

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QDialog
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QTableWidget, QLineEdit, QLabel, QPushButton, QTableWidgetItem,\
    QCheckBox
from PySide6.QtGui import QAction, QImage, QBrush, QColor
from PySide6.QtCore import Qt, QSize, Slot, QThread

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
os.chdir(ROOT)

from core.bbox import BBox
from utils.config import _C as cfg, update_config
from utils.logger import init_logger, get_logger

from core.obj_detectors import ObjectDetector
from core.tracking import ObjectTracker
from utils.medialoader import MediaLoader

from gui.image import ImgDialog


ROW = OrderedDict({"chicken": "닭고기류",
                   "sauce1": "소스 1",
                   "sauce2": "소스 2",
                   "powder": "파우더",
                   "quantity": "발주물량",
                   "match": "집계물량"})
PROD = [
    {
        "name": "제품 1",
        "chicken": ["닭윙1.3kg", 50],
        "sauce1": ["닭갈비간장500g", 25],
        "sauce2": ["볼케이노소스500g", 25],
        "powder": ["시즈닝100g", 50],
        "quantity": 50,
    },
    {
        "name": "제품 2",
        "chicken": ["닭볶음용1.3kg", 30],
        "sauce1": ["허니버터소스1kg", 15],
        "sauce2": [],
        "powder": ["시즈닝500g", 30],
        "quantity": 30,
    },
    {
        "name": "제품 3",
        "chicken": ["닭다리살1kg", 50],
        "sauce1": ["소떡소떡소스1kg", 50],
        "sauce2": [],
        "powder": ["치킨파우더500g", 50],
        "quantity": 50,
    },
    {
        "name": "제품 4",
        "chicken": ["닭볶음탕용600g", 50],
        "sauce1": ["갈릭디핑소스500g", 50],
        "sauce2": [],
        "powder": ["뿌링클링시즈닝150g", 50],
        "quantity": 50,
    },
    {
        "name": "제품 5",
        "quantity": 50,
        "chicken": ["훈제오리400g", 50],
        "sauce1": ["치즈치폴레소스500g", 25],
        "sauce2": ["닭갈비간장500g", 25],
        "powder": ["뿌링클링시즈닝150g", 50]
    },
]
PROD_CLASS = {
    "태음융융소금염지닭": 0,
    "닭윙1.3kg": 1,
    "시즈닝100g": 2,
    "시즈닝500g": 3,
    "닭볶음용1.3kg": 4,
    "닭다리살1kg": 5,
    "닭갈비간장500g": 6,
    "닭볶음탕용600g": 7,
    "허니버터소스1kg": 8,
    "소떡소떡소스1kg": 9,
    "닭갈비고추장500g": 10,
    "갈릭디핑소스500g": 11,
    "치즈치폴레소스500g": 12,
    "훈제오리400g": 13,
    "오리안심500g": 14,
    "치킨파우더500g": 15,
    "볼케이노소스500g": 16,
    "뿌링클링시즈닝150g": 17,
}


def getClassNumberDict(cls_tbl, prod_tbl, column: list):
    gt_table = defaultdict(int)
    for prod in prod_tbl:
        for c in column:
            if prod[c]:
                gt_table[cls_tbl[prod[c][0]]] += prod[c][1]
    return gt_table


def getKeybyValue(cls_dict, idx):
    for k, v in cls_dict.items():
        if v == idx:
            return k


class AnalysisThread(QThread):
    def __init__(self, parent=None, medialoader=None, statusbar=None, detector=None, tracker=None, img_viewer=None,
                 logger=None):
        super().__init__(parent=parent)
        self.cfg = parent.config

        self.medialoader = medialoader if medialoader is not None else self.parent().medialoader
        self.sbar = statusbar if statusbar is not None else self.parent().sbar
        self.detector = detector if detector else self.parent().obj_detector
        self.tracker = tracker if tracker else self.parent().obj_tracker
        self.viewer = img_viewer if img_viewer else self.parent().live_viewer

        self.logger = logger if logger is not None else get_logger()
        self.log_interval = self.parent().config.CONSOLE_LOG_INTERVAL

        self.stop = False

        # analysis logging
        self.f_cnt = 0
        self.ts = [0., 0., 0.,]
        self.class_cnt = defaultdict(int)
        self.id_cnt = defaultdict(int)

        self.table_widget = self.parent().middle
        self.gt_cls_num = self.table_widget.gt_cls_num
        self.prod_loc = self.table_widget.content_loc

        self.over_cnt = defaultdict(int)
        self.less_cnt = defaultdict(int)

    def run(self):
        self.logger.info("영상 분석 쓰레드 시작")
        self.sbar.showMessage("영상 분석 시작")

        self.medialoader.restart()
        self.stop = False

        # detection filter area set
        f = self.medialoader.wait_frame()
        img_h, img_w = f.shape[:2]
        filter_ratio = 0.1
        filter_x1, filter_y1 = int(img_w * filter_ratio), int(img_h * filter_ratio)
        filter_x2, filter_y2 = int(img_w * (1 - filter_ratio)), int(img_h * (1 - filter_ratio))

        while True:
            frame = self.medialoader.get_frame()
            if frame is None or self.stop is True:
                break

            t0 = time.time()
            im = self.detector.preprocess(frame)
            _pred = self.detector.detect(im)
            _pred, _det = self.detector.postprocess(_pred)

            # box filtering
            _dets = []
            _boxes = []
            for _d in _det:
                if filter_x1 < (_d[0] + _d[2]) / 2 < filter_x2 and filter_y1 < (_d[1] + _d[3]) / 2 < filter_y2:
                    _dets.append(_d)
                    _boxes.append(BBox(tlbr=_d[:4], class_index=int(_d[5]),
                                       class_name=self.detector.names[_d[5]], conf=_d[4], imgsz=frame.shape))
            _det = np.array(_dets)
            t1 = time.time()

            # tracking update
            if len(_det):
                track_ret = self.tracker.update(_det, frame)
                if len(track_ret):
                    t_boxes = track_ret[:, 0:4].astype(np.int32)
                    t_ids = track_ret[:, 4].astype(np.int32)
                    t_confs = track_ret[:, 5]
                    t_classes = track_ret[:, 6]
                    for i, (xyxy, _id, conf, cls) in enumerate(zip(t_boxes, t_ids, t_confs, t_classes)):
                        _boxes[i].tracking_id = _id
            t2 = time.time()

            for _box in _boxes:
                t_id = _box.tracking_id
                if t_id != -1:
                    self.id_cnt[t_id] += 1
                    if self.id_cnt[t_id] == 5:
                        self.class_cnt[_box.class_idx] += 1

                        self.edit_prod_num(_box.class_idx)
                        self.logger.info(f"add 1 prod {_box.class_idx} - {getKeybyValue(PROD_CLASS, _box.class_idx)}")

            # visualize
            if self.parent().ck_liveView.isChecked():
                if filter_ratio != 0:
                    cv2.rectangle(frame, (filter_x1, filter_y1), (filter_x2, filter_y2),
                                  color=(96, 216, 96),
                                  thickness=2, lineType=cv2.LINE_AA)

                for b in _boxes:
                    if b.tracking_id != -1:
                        cv2.rectangle(frame, (b.x1, b.y1), (b.x2, b.y2), (96, 96, 216), thickness=2, lineType=cv2.LINE_AA)
                        font = ImageFont.truetype('./data/fonts/NanumMyeongjoEcoBold.ttf', 16)
                        img_pil = Image.fromarray(frame)
                        img_draw = ImageDraw.Draw(img_pil)
                        img_draw.text((b.x1, b.y1 + 7), f"({b.class_name})ID: {b.tracking_id}", font=font, fill=(216, 96, 96))
                        # cv2.putText(frame, f"({b.class_name})ID: {b.tracking_id}", (b.x1, b.y1 + 10),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (216, 96, 96), 2)
                        frame = np.array(img_pil)

                # for i, (k, v) in enumerate(self.class_cnt.items()):
                #     cv2.putText(frame, f"{k}({self.detector.names[k]}): {v}", (img_w - 150, 30 + i * 30),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (32, 32, 32), 2)

                img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format.Format_BGR888)
                self.viewer.set_image(img)
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

            time.sleep(0.001)

        # cv2.destroyAllWindows()
        self.sbar.showMessage("영상 분석 종료")
        get_logger().info("영상 분석 종료")

    def edit_prod_num(self, class_idx):
        prod_cls = class_idx
        prod_name = getKeybyValue(PROD_CLASS, prod_cls)

        if prod_name in self.prod_loc:
            is_accept = False
            for p_loc in self.prod_loc[prod_name]:
                gt_num = int(self.table_widget.item(p_loc[0], p_loc[1]-1).text())
                prod_num = int(self.table_widget.item(p_loc[0], p_loc[1]).text())

                # 제품 갯수 수정
                if gt_num > prod_num:
                    prod_num += 1
                    item = self.table_widget.item(p_loc[0], p_loc[1])
                    item.setText(str(prod_num))
                    if prod_num == gt_num:      # 목표 수량 도달
                        item.setForeground(QBrush(QColor(0, 0, 255)))
                    is_accept = True

                    # 일치여부 확인 및 갱신
                    self.table_widget.check_matching(p_loc[1] // 3)
                    break

            # 어떤 항목에도 갯수 수정을 하지 못함 -> 초과수량
            if is_accept is False:
                self.over_cnt[prod_name] += 1
                t = "초과 수량: "
                for k, v in self.over_cnt.items():
                    t += f"{k} - {v}EA, "
                self.parent().lb_over.setText(t)

        else:
            print(prod_name, prod_cls, "제품리스트에 없음")


class ProdTable(QTableWidget):
    def __init__(self, col, row):
        super().__init__()
        self.col = col
        self.row = row

        self.gt_cls_num = getClassNumberDict(PROD_CLASS, prod_tbl=col, column=list(row.keys())[:4])

        self.content_loc = {}
        for c, col in enumerate(self.col):
            for r, k in enumerate(list(self.row.keys())[:4]):
                if col[k]:
                    # 집계 물량 테이블 좌표 저장
                    if col[k][0] in list(self.content_loc.keys()):
                        self.content_loc[col[k][0]].append((r+1, 3*c+2, col[k][1]))
                    else:
                        self.content_loc[col[k][0]] = [(r+1, 3*c+2, col[k][1])]

    def set_header(self):
        # Set vertical header
        self.setRowCount(1 + len(self.row))
        self.setVerticalHeaderLabels([""] + list(self.row.values()))

        # Set Horizontal header
        self.horizontalHeader().setVisible(False)
        self.setColumnCount(len(self.col) * 3)
        for i, col in enumerate(self.col):
            self.setSpan(0, 3 * i, 1, 3)
            self.setItem(0, 3 * i, QTableWidgetItem(col["name"]))
            self.item(0, 3 * i).setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            # for _i in range(3):
            #     self.horizontalHeader().setSectionResizeMode(i * 3 + _i,  )

        # Set span not prod
        for i in range(len(self.col)):
            self.setSpan(5, 3 * i, 1, 3)
            self.setSpan(6, 3 * i, 1, 3)

    def set_contents(self):
        for c, col in enumerate(self.col):
            for r, (k, v) in enumerate(list(self.row.items())[:5]):
                if k == "quantity":
                    self.setItem(r + 1, 3 * c, QTableWidgetItem(str(col[k])))
                    self.item(r + 1, 3 * c).setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    continue

                if col[k]:
                    self.setItem(r + 1, 3 * c, QTableWidgetItem(col[k][0]))
                    self.setItem(r + 1, 3 * c + 1, QTableWidgetItem(str(col[k][1])))
                    item = QTableWidgetItem(str(0))
                    item.setForeground(QBrush(QColor(255, 0, 0)))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.setItem(r + 1, 3 * c + 2, item)
                else:
                    self.setItem(r + 1, 3 * c, QTableWidgetItem("-"))

        self.resizeColumnsToContents()

    def show_table(self):
        self.set_header()
        self.set_contents()

    def clear_table(self):
        self.clear()

        # Delete each rows
        r_cnt = self.rowCount()
        for r in range(r_cnt):
            self.removeRow(r_cnt - r - 1)

    def init_result(self):
        r = self.rowCount()
        c = self.columnCount()
        for idx in range(c):
            if c % 3 == 0:
                item = QTableWidgetItem("불일치")
                item.setForeground(QBrush(QColor(255, 0, 0)))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.setItem(r-1, idx, item)

    def check_matching(self, p_num):
        for r in range(1, self.rowCount()):
            gt_num = self.item(r, p_num * 3 + 1)
            if gt_num is not None:
                gt_num = gt_num.text()
                prod_num = self.item(r, p_num * 3 + 2).text()

                if gt_num != prod_num:
                    return

        # matching True
        item = QTableWidgetItem("일치")
        item.setForeground(QBrush(QColor(0, 0, 255)))
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setItem(self.rowCount()-1, p_num * 3, item)

    def get_shortfall(self, cur_dict):
        less_cnt = defaultdict(int)
        for gk, gv in self.gt_cls_num.items():
            if gk in cur_dict.keys():
                if gv - cur_dict[gk] > 0:
                    less_cnt[getKeybyValue(PROD_CLASS, gk)] = gv - cur_dict[gk]
            else:
                less_cnt[getKeybyValue(PROD_CLASS, gk)] = gv
        return less_cnt


class MainWidget(QWidget):
    def __init__(self, cfg, parent=None):
        super().__init__(parent=parent)

        # init MainWidget
        self.config = cfg
        self.logger = get_logger()
        self.logger.info("Create Main QWidget - start")
        self.sbar = self.parent().statusBar()

        # top - layout
        self.top = QHBoxLayout()
        self.el_date = QLineEdit()
        self.el_client = QLineEdit()
        self.el_prod_name = QLineEdit()
        self.bt_search = QPushButton("발주 조회")
        self.bt_reset = QPushButton("초기화")
        self.ck_liveView = QCheckBox("실시간 화면", self)

        self.top.addWidget(QLabel("날짜:"))
        self.top.addWidget(self.el_date)
        self.top.addWidget(QLabel("발주사:"))
        self.top.addWidget(self.el_client)
        self.top.addWidget(QLabel("제품명:"))
        self.top.addWidget(self.el_prod_name)
        self.top.addWidget(self.bt_search)
        self.top.addWidget(self.bt_reset)
        self.top.addWidget(self.ck_liveView)

        # middle - table
        self.middle = ProdTable(col=PROD, row=ROW)

        # bottom - layout
        self.prebotttom = QHBoxLayout()
        self.lb_over = QLabel("분석 결과:", parent=self)
        self.prebotttom.addWidget(self.lb_over)
        self.bottom = QHBoxLayout()
        self.bt_start = QPushButton("수량 학인 시작")
        self.bt_stop = QPushButton("수량 확인 종료")

        self.bt_start.setDisabled(True)
        self.bt_stop.setDisabled(True)

        self.bottom.addWidget(self.bt_start)
        self.bottom.addWidget(self.bt_stop)

        # Main Widget
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        self.main_layout.addLayout(self.top)
        self.main_layout.addWidget(self.middle)
        self.main_layout.addLayout(self.prebotttom)
        self.main_layout.addLayout(self.bottom)

        # analysis image
        self.live_viewer = ImgDialog(parent=self, title="제품분석 - 실시간화면")

        # Set Input Loader
        try:
            self.medialoader = MediaLoader(source=cfg.MEDIA_SOURCE, logger=self.logger, opt=cfg)
            self.medialoader.start()
            self.medialoader.pause()
        except Exception as e:
            self.logger.error(f"Medialoader can't create: {e}")
            self.medialoader = None
        self.obj_detector = ObjectDetector(cfg=cfg)
        self.obj_tracker = ObjectTracker(cfg=cfg)
        self.analysis_thread = None

        # signals & slots
        self.bt_search.clicked.connect(self.setUI)
        self.bt_reset.clicked.connect(self.clearUI)
        self.bt_start.clicked.connect(self.start_analysis)
        self.bt_stop.clicked.connect(self.stop_analysis)

        self.sbar.showMessage("프로그램 준비 완료")
        self.logger.info("Create Main QWidget - end")

    @Slot()
    def setUI(self):
        self.logger.info("setUI - start")
        self.middle.show_table()

        if self.analysis_thread is not None:
            del self.analysis_thread
        self.analysis_thread = AnalysisThread(parent=self,
                                              medialoader=self.medialoader,
                                              statusbar=self.sbar,
                                              detector=self.obj_detector,
                                              tracker=self.obj_tracker,
                                              img_viewer=self.live_viewer)

        self.lb_over.setText("분석 결과:")
        # 일치여부 표시
        self.middle.init_result()

        self.bt_search.setDisabled(True)
        self.bt_start.setDisabled(False)
        self.bt_stop.setDisabled(False)

        self.sbar.showMessage("발주 내용 조회")
        self.logger.info("setUI - end")

    @Slot()
    def clearUI(self):
        self.logger.info("clearUI - start")
        self.middle.clear_table()

        self.lb_over.setText("분석 결과:")

        self.bt_search.setDisabled(False)
        self.bt_start.setDisabled(True)
        self.bt_start.setText("수량 확인 시작")
        self.bt_stop.setDisabled(True)
        self.sbar.showMessage("시스템 초기화")
        self.logger.info("clearUI - end")

    @Slot()
    def start_analysis(self):
        if self.medialoader:
            self.logger.info("MainWidget - Start analysis")
            self.analysis_thread.start()

            # 실시간 화면 표시
            if self.ck_liveView.isChecked():
                self.live_viewer.show()
        else:
            try:
                self.medialoader = MediaLoader(source="0", logger=self.logger)
                self.medialoader.start()

                self.analysis_thread.medialoader = self.medialoader
                time.sleep(1)
                self.logger.info("Medialoader is None and recreate, Start analysis")
                self.analysis_thread.start()

                # 실시간 화면 표시
                if self.ck_liveView.isChecked():
                    self.live_viewer.show()
            except Exception as e:
                # Exception Camera Connection
                self.logger.error(f"Camera is not Connected: {e}")
                _dialog = QDialog(parent=self)
                _alert = QWidget(self)
                _layout = QHBoxLayout()
                _label = QLabel("카메라가 연결되어 있지 않습니다. 연결을 확인해주세요.")
                _layout.addWidget(_label)
                _dialog.setLayout(_layout)
                _dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
                _dialog.show()

    @Slot()
    def stop_analysis(self):
        self.logger.info("MainWidget - Stop analysis")

        self.bt_start.setText("수량 확인 재개")
        
        self.analysis_thread.stop = True
        self.analysis_thread.exit()

        self.live_viewer.hide()

        self.analysis_thread.less_cnt = self.middle.get_shortfall(self.analysis_thread.class_cnt)
        self.show_result()

    def show_result(self):
        t = "초과 수량: "
        for i, (k, v) in enumerate(self.analysis_thread.over_cnt.items()):
            t += f"{k} - {v}EA, "
            if (i + 1) % 6 == 0:
                t += '\n'
        t += "\n부족 수량: "
        for i, (k, v) in enumerate(self.analysis_thread.less_cnt.items()):
            t += f"{k} - {v}EA, "
            if (i + 1) % 6 == 0:
                t += '\n'
        self.lb_over.setText(t)


class DaolCND(QMainWindow):
    def __init__(self, config=None):
        super().__init__()
        # init gui
        self.logger = get_logger()
        self.logger.info("Init DaolCND QMainWindow - start")

        # resize window
        self.geo = self.screen().availableGeometry()
        self.resize(QSize(self.geo.width() * 0.5, self.geo.height() * 0.5))

        # MenuBar
        self.mbar = self.menuBar()
        self.file_menu = self.mbar.addMenu("File")

        # Menu - Exit QAction
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        self.file_menu.addAction(exit_action)

        # Status Bar
        self.sbar = self.statusBar()
        self.sbar.showMessage("프로그램 준비 시작", 0)

        # Set Central Widget
        self.main_widget = MainWidget(cfg=config, parent=self)
        self.setCentralWidget(self.main_widget)

        self.logger.info("Init DaolCND QMainWindow - end")


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='configuration', default='./configs/dv1.yaml')
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

    app_gui = DaolCND(cfg)
    app_gui.setWindowTitle("다올씨앤디")
    app_gui.show()

    sys.exit(app.exec())
