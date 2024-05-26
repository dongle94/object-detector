import os
import sys
import time
import copy
from collections import OrderedDict, namedtuple

import numpy as np
from typing import Union
from pathlib import Path

import tensorrt as trt
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.yolov5 import YOLOV5
from core.yolov5.yolov5_utils.torch_utils import select_device
from core.yolov5.yolov5_utils.augmentations import letterbox


class Yolov5TRT(YOLOV5):
    def __init__(self, weight: str, device: str = "cpu", img_size: int = 640, fp16: bool = False, auto: bool = False,
                 gpu_num: int = 0, conf_thres=0.25, iou_thres=0.45, agnostic=False, max_det=100,
                 classes: Union[list, None] = None):
        super(Yolov5TRT, self).__init__()
        self.device = select_device(device=device, gpu_num=gpu_num)
        self.img_size = img_size
        self.gpu_num = gpu_num
        self.fp16 = True if fp16 is True else False

        self.trt_logger = trt.Logger(trt.Logger.INFO)

        with open(weight, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            self.model = runtime.deserialize_cuda_engine(f.read())
        self.context = self.model.create_execution_context()

        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        self.bindings = OrderedDict()
        self.output_names = []
        for i in range(self.model.num_bindings):
            name = self.model.get_binding_name(i)
            dtype = trt.nptype(self.model.get_binding_dtype(i))
            if 'onnx::' in name:
                continue
            if not self.model.binding_is_input(i):
                self.output_names.append(name)
            shape = tuple(self.context.get_binding_shape(i))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.batch_size = self.bindings['images'].shape[0]

        # prams for postprocessing
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic = agnostic
        self.max_det = max_det

    def warmup(self, img_size=(1, 3, 640, 640)):
        #TODO CHECK
        im = torch.empty(*img_size, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
        t = self.get_time()
        self.infer(im)  # warmup
        print(f"-- Yolov5 Detector warmup: {self.get_time() - t:.6f} sec --")

    def preprocess(self, img):
        # TODO CHECK
        im = letterbox(img, new_shape=self.img_size, auto=self.auto, stride=self.stride)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = torch.unsqueeze(im, dim=0)  # expand for batch dim
        return im, img

    def infer(self, img):
        self.binding_addrs['images'] = int(img.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = [self.bindings[x].data for x in sorted(self.output_names)]
        print(y)
        return y

    def postprocess(self, pred, im_shape, im0_shape):
        pass

    def get_time(self):
        return time.time()


if __name__ == "__main__":
    import cv2
    from utils.config import set_config, get_config

    set_config('./configs/config.yaml')
    cfg = get_config()

    yolov5 = Yolov5TRT(
        cfg.det_model_path, device=cfg.device, img_size=cfg.yolov5_img_size, fp16=cfg.det_half, auto=False,
        gpu_num=cfg.gpu_num, conf_thres=cfg.det_conf_thres, iou_thres=cfg.yolov5_nms_iou,
        agnostic=cfg.yolov5_agnostic_nms, max_det=cfg.yolov5_max_det, classes=cfg.det_obj_classes
    )
    yolov5.warmup(img_size=(1, 3, cfg.yolov5_img_size, cfg.yolov5_img_size))

    _im = cv2.imread('./data/images/sample.jpg')
    t0 = yolov5.get_time()
    _im, _im0 = yolov5.preprocess(_im)
    t1 = yolov5.get_time()
    _y = yolov5.infer(_im)
    t2 = yolov5.get_time()
