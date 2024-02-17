import os
import sys
import time
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.logger import get_logger


class ObjectDetector(object):
    def __init__(self, cfg=None):
        self.logger = get_logger(cfg.logger_name)
        self.cfg = cfg

        weight = os.path.abspath(cfg.det_model_path)
        self.detector_type = cfg.det_model_type.lower()

        device = cfg.device
        gpu_num = cfg.gpu_num
        fp16 = cfg.det_half
        conf_thres = cfg.det_conf_thres
        classes = cfg.det_obj_classes

        if self.detector_type == "yolov5":
            img_size = cfg.yolov5_img_size
            iou_thres = cfg.yolov5_nms_iou
            agnostic = cfg.yolov5_agnostic_nms
            max_det = cfg.yolov5_max_det
            self.im_shape = None
            self.im0_shape = None

            # model load with weight
            ext = os.path.splitext(weight)[1]
            if ext in ['.pt', '.pth']:
                from core.yolov5.yolov5_pt import Yolov5Torch
                model = Yolov5Torch
            elif ext == '.onnx':
                from core.yolov5.yolov5_ort import Yolov5ORT
                model = Yolov5ORT
            else:
                raise FileNotFoundError('No Yolov5 weight File!')
            self.detector = model(
                weight=weight,
                device=device,
                img_size=img_size,
                fp16=fp16,
                gpu_num=gpu_num,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                agnostic=agnostic,
                max_det=max_det,
                classes=classes
            )
            self.names = self.detector.names

            # warm up
            self.detector.warmup(imgsz=(1, 3, img_size, img_size))
            self.logger.info(f"Successfully loaded weight from {weight}")

            # logging
            self.f_cnt = 0
            self.ts = [0., 0., 0.]

    def run(self, img):
        if self.detector_type == "yolov5":
            t0 = self.detector.get_time()

            img, orig_img = self.detector.preprocess(img)
            im_shape = img.shape
            im0_shape = orig_img.shape
            t1 = self.detector.get_time()

            preds = self.detector.infer(img)
            t2 = self.detector.get_time()

            pred, det = self.detector.postprocess(preds, im_shape, im0_shape)
            t3 = self.detector.get_time()

            # calculate time & logging
            self.f_cnt += 1
            self.ts[0] += t1 - t0
            self.ts[1] += t2 - t1
            self.ts[2] += t3 - t2
            if self.f_cnt % self.cfg.console_log_interval == 0:
                self.logger.debug(
                    f"{self.detector_type} detector {self.f_cnt} Frames average time - "
                    f"preproc: {self.ts[0]/self.f_cnt:.6f} sec / "
                    f"infer: {self.ts[1] / self.f_cnt:.6f} sec / " 
                    f"postproc: {self.ts[2] / self.f_cnt:.6f} sec")

        else:
            pred, det = None, None

        return det


if __name__ == "__main__":
    import cv2
    from utils.medialoader import MediaLoader
    from utils.logger import init_logger
    from utils.config import set_config, get_config

    set_config('./configs/config.yaml')
    _cfg = get_config()

    init_logger(_cfg)
    _logger = get_logger()

    _detector = ObjectDetector(cfg=_cfg)

    s = sys.argv[1]
    media_loader = MediaLoader(s, realtime=True)
    media_loader.start()

    while media_loader.is_frame_ready() is False:
        time.sleep(0.01)
        continue

    while True:
        frame = media_loader.get_frame()

        _det = _detector.run(frame)

        for d in _det:
            x1, y1, x2, y2 = map(int, d[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (96, 96, 216), thickness=2, lineType=cv2.LINE_AA)

        cv2.imshow('_', frame)
        if cv2.waitKey(1) == ord('q'):
            print("-- CV2 Stop --")
            break

    media_loader.stop()
    print("-- Stop program --")