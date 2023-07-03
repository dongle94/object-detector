import os
import sys

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
os.chdir(ROOT)

from obj_detectors.yolov5_pt import YoloDetector


class ObjectDetector(object):
    def __init__(self, cfg=None):
        # Detection model configuration
        if os.path.abspath(cfg.DET_MODEL_PATH) != cfg.DET_MODEL_PATH:
            weight = os.path.abspath(os.path.join(ROOT, cfg.DET_MODEL_PATH))
        else:
            weight = os.path.abspath(cfg.DET_MODEL_PATH)
        self.detector_type = cfg.DET_MODEL_TYPE.lower()

        if self.detector_type == "yolo":
            device = cfg.DEVICE
            fp16 = cfg.HALF
            img_size = cfg.IMG_SIZE
            self.im_shape = None
            self.im0_shape = None

            # model load with weight
            self.detector = YoloDetector(weight=weight, device=device, img_size=img_size, fp16=fp16)

            # warm up
            self.detector.warmup(imgsz=(1, 3, img_size, img_size))

    def preprocess(self, img):
        if self.detector_type == "yolo":
            img, orig_img = self.detector.preprocess(img)
            self.im_shape = img.shape
            self.im0_shape = orig_img.shape

        return img

    def detect(self, img):
        preds = None
        if self.detector_type == "yolo":
            preds = self.detector.forward(img)

        return preds

    def postprocess(self, ret):
        preds, dets = None, None
        if self.detector_type == 'yolo':
            max_det = 100
            preds, dets = self.detector.postprocess(pred=ret, im_shape=self.im_shape,
                                                    im0_shape=self.im0_shape, max_det=max_det)
            preds = preds.cpu().numpy()
            dets = dets.cpu().numpy()

        return preds, dets


if __name__ == "__main__":
    import time
    import cv2
    from utils.config import _C as cfg
    from utils.config import update_config
    from utils.medialoader import MediaLoader
    from utils.logger import get_logger, init_logger

    update_config(cfg, args='./config.yaml')
    init_logger(cfg)
    logger = get_logger()

    detector = ObjectDetector(cfg=cfg)

    s = sys.argv[1]
    media_loader = MediaLoader(s)
    time.sleep(1)
    f_cnt = 0
    ts = [0., 0., 0.]
    while True:
        frame = media_loader.get_frame()

        t0 = time.time()
        im = detector.preprocess(frame)
        t1 = time.time()
        _pred = detector.detect(im)
        t2 = time.time()
        _pred, _det = detector.postprocess(_pred)
        t3 = time.time()

        ts[0] += (t1 - t0)
        ts[1] += (t2 - t1)
        ts[2] += (t3 - t2)

        for d in _det:
            x1, y1, x2, y2 = map(int, d[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (96, 96, 216), thickness=2, lineType=cv2.LINE_AA)

        cv2.imshow('_', frame)
        if cv2.waitKey(1) == ord('q'):
            print("-- CV2 Stop --")
            break

        f_cnt += 1
        if f_cnt % 10 == 0:
            logger.debug(f"{f_cnt} Frame - pre: {ts[0]/f_cnt:.4f} / infer: {ts[1]/f_cnt:.4f} / post: {ts[2]/f_cnt:.4f}")

    media_loader.stop()
    print("-- Stop program --")
