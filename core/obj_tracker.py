import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.logger import get_logger


class ObjectTracker(object):
    def __init__(self, cfg):
        weight = os.path.abspath(cfg.track_model_path)
        self.tracker_type = cfg.track_model_type.lower()

        if self.tracker_type == 'deepocsort':
            from core.tracking.deepocsort import DeepOCSort
            device = cfg.device
            fp16 = cfg.track_half
            embedding_off = not cfg.track_use_encoder

            self.tracker = DeepOCSort(
                model_weights=Path(weight),
                device=device,
                fp16=fp16,
                embedding_off=embedding_off
            )

    def update(self, dets, img, tag='bulb'):
        if self.tracker_type == 'deepocsort':
            return self.tracker.update(dets, img, tag)
        else:
            return None


if __name__ == "__main__":
    import time
    import cv2
    import numpy as np

    from utils.config import set_config, get_config
    from utils.logger import init_logger

    from core.obj_detector import ObjectDetector
    from core.media_loader import MediaLoader
    from core.bbox import BBox

    set_config('./configs/config.yaml')
    _cfg = get_config()

    init_logger(_cfg)
    _logger = get_logger()

    _detector = ObjectDetector(cfg=_cfg)
    _tracker = ObjectTracker(cfg=_cfg)

    s = sys.argv[1]
    media_loader = MediaLoader(s, realtime=0, bgr=True)

    f_cnt = 0
    ts = [0, 0, ]
    while True:
        frame = media_loader.get_frame()

        img_h, img_w = frame.shape[:2]
        filter_ratio = 0.2
        filter_x1, filter_y1 = int(img_w * filter_ratio), int(img_h * filter_ratio)
        filter_x2, filter_y2 = int(img_w * (1 - filter_ratio)), int(img_h * (1 - filter_ratio))

        t0 = time.time()
        _det = _detector.run_np(frame)

        # box filtering
        _dets = []
        _boxes = []
        for _d in _det:
            if filter_x1 < (_d[0] + _d[2]) / 2 < filter_x2 and filter_y1 < (_d[1] + _d[3]) / 2 < filter_y2:
                _dets.append(_d)
                _boxes.append(BBox(tlbr=_d[:4], class_name=_detector.names[_d[5]], conf=_d[4], imgsz=frame.shape))
        _det = np.array(_dets)
        t1 = time.time()

        # tracking update
        if len(_det):
            track_ret = _tracker.update(_det, frame)
            if len(track_ret):
                t_boxes = track_ret[:, 0:4].astype(np.int32)
                t_ids = track_ret[:, 4].astype(np.int32)
                t_confs = track_ret[:, 5]
                t_classes = track_ret[:, 6]
                for i, (xyxy, _id, conf, cls) in enumerate(zip(t_boxes, t_ids, t_confs, t_classes)):
                    _boxes[i].tracking_id = _id
                    # tracking visualize
                    im = cv2.rectangle(frame,
                                       (xyxy[0] + 5, xyxy[1] + 5), (xyxy[2] - 5, xyxy[3] - 5),
                                       (216, 96, 96), 2)
                    cv2.putText(frame, f'id: {_id}', (xyxy[0], xyxy[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (216, 96, 96), 2)
        t2 = time.time()

        # calculate time
        ts[0] += (t1 - t0)
        ts[1] += (t2 - t1)

        cv2.rectangle(frame, (filter_x1, filter_y1), (filter_x2, filter_y2),
                      color=(96, 216, 96),
                      thickness=2, lineType=cv2.LINE_AA)
        for d in _det:
            x1, y1, x2, y2 = map(int, d[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (96, 96, 216), thickness=2, lineType=cv2.LINE_AA)

        cv2.imshow('_', frame)
        if cv2.waitKey(1) == ord('q'):
            print("-- CV2 Stop --")
            break

        f_cnt += 1
        if f_cnt % _cfg.console_log_interval == 0:
            _logger.debug(
                f"{f_cnt} Frame - det: {ts[0] / f_cnt:.4f} / tracking: {ts[1] / f_cnt:.4f}")
