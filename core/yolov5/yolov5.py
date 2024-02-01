import copy
import numpy as np
from typing import Union

import torch.cuda

from core.yolov5 import YOLOV5
from core.yolov5.yolov5_utils.torch_utils import select_device
from core.yolov5.yolov5_utils.general import check_img_size, non_max_suppression, scale_boxes
from core.yolov5.yolov5_utils.augmentations import letterbox
from core.yolov5.models.experimental import attempt_load


class Yolov5Torch(YOLOV5):
    def __init__(self, weight: str, device: str = "cpu", img_size: int = 640, fp16: bool = False, auto: bool = False,
                 fuse: bool = True, gpu_num: int = 0, conf_thres=0.25, iou_thres=0.45, agnostic=False, max_det=100,
                 classes: Union[list, None] = None):
        super().__init__()
        self.device = select_device(device=device, gpu_num=gpu_num)
        self.cuda = torch.cuda.is_available() and device != "cpu"
        self.fp16 = True if fp16 is True and self.device.type != "cpu" else False
        model = attempt_load(weight, device=self.device, inplace=True, fuse=fuse)
        model.half() if self.fp16 else model.float()
        self.model = model
        self.stride = max(int(model.stride.max()), 32)
        self.img_size = check_img_size(img_size, s=self.stride)
        self.names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        self.auto = auto

        # parameter for postprocessing
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic = agnostic
        self.max_det = max_det

    def warmup(self, imgsz=(1, 3, 640, 640)):
        im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)
        print("-- Yolov5 Detector warmup --")

    def preprocess(self, img: np.ndarray):
        im = letterbox(img, new_shape=self.img_size, auto=self.auto, stride=self.stride)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = torch.unsqueeze(im, dim=0)  # expand for batch dim
        return im, img

    def infer(self, im):
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()
        y = self.model(im)
        return y

    def postprocess(self, pred, im_shape, im0_shape):
        pred = non_max_suppression(pred,
                                   conf_thres=self.conf_thres,
                                   iou_thres=self.iou_thres,
                                   classes=self.classes,
                                   agnostic=self.agnostic,
                                   max_det=self.max_det)[0]
        det = scale_boxes(im_shape[2:], copy.deepcopy(pred[:, :4]), im0_shape).round()
        det = torch.cat([det, pred[:, 4:]], dim=1)

        return pred, det


if __name__ == "__main__":
    import numpy as np
    yolov5 = Yolov5Torch("./weights/yolov5m.pt", device='mps', fp16=True, auto=False)
    yolov5.warmup()

    _im = np.zeros((1280, 720, 3), dtype=np.uint8)
    _im, _im0 = yolov5.preprocess(_im)
    y = yolov5.infer(_im)
    y = yolov5.postprocess(y, _im.size(), _im0.shape)