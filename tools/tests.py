import os
import sys
import cv2
import numpy as np
import dlib

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
os.chdir(ROOT)

from utils.config import _C as cfg
from utils.config import update_config
from utils.logger import get_logger, init_logger
from utils.medialoader import MediaLoader

from core.obj_detectors import ObjectDetector
from core.tracking import ObjectTracker
from core.bbox import BBox
from core.gender_classification import GenderModel



if __name__ == "__main__":
    import time

    update_config(cfg, args='./config.yaml')
    init_logger(cfg)
    logger = get_logger()

    detector = ObjectDetector(cfg=cfg)
    tracker = ObjectTracker(cfg=cfg)
    face_detector = dlib.cnn_face_detection_model_v1('./weights/mmod_human_face_detector.dat')
    gender_classifier = GenderModel()

    # media_loader = MediaLoader('/home/dongle94/Videos/MOT17-04.webm', realtime=False)
    media_loader = MediaLoader('/home/dongle94/Videos/MOT17-09.webm', realtime=False)
    # media_loader = MediaLoader('/home/dongle94/Videos/MOT17-11.webm', realtime=False)
    media_loader.start()

    while media_loader.is_frame_ready() is False:
        time.sleep(0.01)
        continue

    f_cnt = 0
    ts = [0., 0., 0.]
    mem_track = {}
    while True:
        frame = media_loader.get_frame()

        t0 = time.time()
        im = detector.preprocess(frame)
        _pred = detector.detect(im)
        _pred, _det = detector.postprocess(_pred)
        t1 = time.time()

        if len(_det):
            track_ret = tracker.update(_det, frame)
            t_boxes = track_ret[:, 0:4].astype(np.int32)
            t_ids = track_ret[:, 4].astype(np.int32)
            t_confs = track_ret[:, 5]
            t_classes = track_ret[:, 6]

            for i, (xyxy, _id) in enumerate(zip(t_boxes, t_ids)):
                if _id not in mem_track.keys():
                    mem_track[_id] = BBox(tlbr=_det[i][:4],
                                          class_name=detector.names[_det[i][5]],
                                          conf=_det[i][4],
                                          imgsz=frame.shape)
                    mem_track[_id].tracking_id = _id
                else:   # already tracking id exist
                    mem_track[_id].set_points(xyxy)

                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (96, 96, 216), thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(frame, f'id: {_id}', (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        t2 = time.time()

        for _d in _det:
            x0, y0, x1, y1 = int(_d[0]), int(_d[1]), int(_d[2]), int(_d[3])
            face_img = frame[y0:y1, x0:x1]
            boxes_face = face_detector(face_img, 2)
            if len(boxes_face) != 0:
                for b in boxes_face:
                    b = b.rect
                    _x0, _y0, _x1, _y1 = b.left(), b.top(), b.right(), b.bottom()
                    cv2.rectangle(frame, (x0+_x0, y0+_y0), (x0+_x1, y0+_y1), (0, 255, 0), 2)
                    # cv2.rectangle(face_img, (_x0, _y0), (_x1, _y1), (0, 255, 0), 2)
                    # cv2.imshow("@@", face_img)
                    # if cv2.waitKey(0) == ord('q'):
                    #     print("-- CV2 Stop --")
                    #     break

        t3 = time.time()

        ts[0] += (t1 - t0)
        ts[1] += (t2 - t1)
        ts[2] += (t3 - t2)

        cv2.imshow('_', frame)
        if cv2.waitKey(0) == ord('q'):
            print("-- CV2 Stop --")
            break

        f_cnt += 1
        if f_cnt % 10 == 0:
            logger.debug(
                f"{f_cnt} Frame - det: {ts[0] / f_cnt:.4f} /"
                f" track: {ts[1] / f_cnt:.4f} /"
                f" face_det: {ts[2] / f_cnt:.4f}")
        # time.sleep(0.5)

    media_loader.stop()
    print("-- Stop program --")