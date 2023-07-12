import os
import sys
import numpy as np

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
os.chdir(ROOT)

from tracking.deepocsort import DeepOCSort


if __name__ == "__main__":
    import time
    import cv2
    from utils.config import _C as cfg
    from utils.config import update_config
    from utils.logger import get_logger, init_logger

    from obj_detectors import ObjectDetector
    from utils.medialoader import MediaLoader

    update_config(cfg, args='./config.yaml')
    init_logger(cfg)
    logger = get_logger()

    detector = ObjectDetector(cfg=cfg)
    tracker = DeepOCSort(
        model_weights=Path("./weights/osnet_x0_25_market1501.pt"),
        device='cuda:0',
        fp16=True
    )

    s = sys.argv[1]
    media_loader = MediaLoader(s)
    media_loader.start()

    while media_loader.is_frame_ready() is False:
        time.sleep(0.01)
        continue

    f_cnt = 0
    ts = [0, 0,]
    while True:
        frame = media_loader.get_frame()

        t0 = time.time()
        im = detector.preprocess(frame)
        _pred = detector.detect(im)
        _pred, _det = detector.postprocess(_pred)
        t1 = time.time()

        track_ret = tracker.update(_det, frame)
        t2 = time.time()

        ts[0] += (t1 - t0)
        ts[1] += (t2 - t1)


        for d in _det:
            x1, y1, x2, y2 = map(int, d[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (96, 96, 216), thickness=2, lineType=cv2.LINE_AA)

        if track_ret.shape[0] != 0:
            t_boxes = track_ret[:, 0:4].astype(np.int32)
            t_ids = track_ret[:, 4].astype(np.int32)
            t_confs = track_ret[:, 5]
            t_classes = track_ret[:, 6]
            for xyxy, _id, conf, cls in zip(t_boxes, t_ids, t_confs, t_classes):
                im = cv2.rectangle(
                    frame,
                    (xyxy[0]+5, xyxy[1]+5),
                    (xyxy[2]-5, xyxy[3]-5),
                    (216, 96, 96),
                    2
                )
                cv2.putText(
                    frame,
                    f'id: {_id}',
                    (xyxy[0], xyxy[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (216, 96, 96),
                    2
                )

        cv2.imshow('_', frame)
        if cv2.waitKey(1) == ord('q'):
            print("-- CV2 Stop --")
            break

        f_cnt += 1
        if f_cnt % 10 == 0:
            logger.debug(
                f"{f_cnt} Frame - det: {ts[0] / f_cnt:.4f} / tracking: {ts[1] / f_cnt:.4f}")
