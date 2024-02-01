import sys
import time
import numpy as np
from typing import Union
from pathlib import Path

import onnxruntime as ort

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.yolov5 import YOLOV5


class Yolov5ORT(YOLOV5):
    def __init__(self, weight: str, device: str = "cpu", img_size: int = 640, fp16: bool = False, auto: bool = False,
                 fuse: bool = True, gpu_num: int = 0, conf_thres=0.25, iou_thres=0.45, agnostic=False, max_det=100,
                 classes: Union[list, None] = None):
        super().__init__()

        # parameter for postprocessing
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic = agnostic
        self.max_det = max_det

    def warmup(self, imgsz=(1, 3, 640, 640)):
        pass

    def preprocess(self, img: np.ndarray):
        pass

    def infer(self, im):
        pass

    def postprocess(self, pred, im_shape, im0_shape):
        pass
