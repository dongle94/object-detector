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

    def run(self, img):
        if self.detector_type == "yolov5":
            img, orig_img = self.detector.preprocess(img)
            im_shape = img.shape
            im0_shape = orig_img.shape

            preds = self.detector.infer(img)
            pred, det = self.detector.postprocess(preds, im_shape, im0_shape)
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

    f_cnt = 0
    t_sum = 0.
    while True:
        frame = media_loader.get_frame()

        st = time.time()
        _det = _detector.run(frame)
        et = time.time()

        t = et - st
        t_sum += t

        for d in _det:
            x1, y1, x2, y2 = map(int, d[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (96, 96, 216), thickness=2, lineType=cv2.LINE_AA)

        cv2.imshow('_', frame)
        if cv2.waitKey(1) == ord('q'):
            print("-- CV2 Stop --")
            break

        f_cnt += 1
        if f_cnt % _cfg.console_log_interval == 0:
            _logger.debug(f"{f_cnt} Frame - run: {t_sum/f_cnt:.4f} sec.")

    media_loader.stop()
    print("-- Stop program --")
