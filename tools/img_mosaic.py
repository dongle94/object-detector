# -*- coding: utf-8 -*-
"""
Image mosaic process
- dlib face detection
- yolo face object detection
"""

import os
import sys
import cv2
import dlib
import argparse
import time

import numpy as np
from tqdm import tqdm


from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
os.chdir(ROOT)

from utils.config import _C as cfg
from utils.config import update_config
from utils.logger import get_logger, init_logger

from core.obj_detectors import ObjectDetector


def mosaic(img: np.ndarray, coord, block=10):
    x1, y1, x2, y2 = coord
    w, h = x2 - x1+1, y2 - y1+1
    if w < block:
        block /= 2
    gap_w, gap_h = int(w/block), int(h/block)
    if gap_w == 0 or gap_h == 0:
        return
    for c in range(3):
        for ny in range(y1, y2, gap_h):
            for nx in range(x1, x2, gap_w):
                new_y = ny + gap_h + 1 if ny + gap_h < y2 else y2
                new_x = nx + gap_w + 1 if nx + gap_w < x2 else x2
                img[ny:new_y, nx:new_x, c] = np.mean(img[ny:new_y, nx:new_x, c])


def process_mosaic(opt):
    inps = opt.inputs
    save_dir = opt.save_path
    is_show = opt.show
    mosaic_size = opt.mosaic_size

    update_config(cfg, args='./configs/mosaic.yaml')
    init_logger(cfg)
    logger = get_logger()

    detector = ObjectDetector(cfg=cfg)
    face_detector = dlib.cnn_face_detection_model_v1('./weights/mmod_human_face_detector.dat')

    # Read dirs
    is_file, is_dir = os.path.isfile(inps), os.path.isdir(inps)
    logger.info(f"{is_file=}, {is_dir=}")
    if save_dir and not os.path.exists(save_dir):
        logger.info(f"Make save dir: {save_dir}")
        os.makedirs(save_dir)

    # Preprocess image list
    if is_file and isinstance(inps, str):
        inps = [os.path.abspath(inps)]
    elif is_dir:
        files = os.listdir(inps)
        inps = [os.path.abspath(os.path.join(inps, f)) for f in files
                if os.path.isfile(os.path.abspath(os.path.join(inps, f)))]
        inps.sort()
    else:
        logger.info("There is no image files")
        return

    # Image process loop Start
    f_cnt = 0
    st = time.time()
    for inp in tqdm(inps, "Processing image mosaic"):
        f0 = cv2.imread(inp, cv2.IMREAD_COLOR)

        if len(f0.shape) == 2:
            f0 = cv2.cvtColor(f0, cv2.COLOR_GRAY2RGB)

        # Detect faces & people by yolo
        im = detector.preprocess(f0)
        _pred = detector.detect(im)
        _pred, _det = detector.postprocess(_pred)

        f1 = f0.copy()

        # Detect faces by dlib
        upsample = 0
        if f0.shape[0] * f0.shape[1] < 640 * 360:
            upsample = 2
        elif f0.shape[0] * f0.shape[1] < 1280 * 720:
            upsample = 1
        whole_det = face_detector(f0, upsample)

        # People boxes 2nd detect faces by dlib
        for d in _det:
            # class == Person
            if int(d[5]) == 0:
                x1, y1, x2, y2 = map(int, d[:4])

                face_img = f0[y1:y2, x1:x2]
                # Process mosaic after detecting face each person
                upsample = 0
                if face_img.shape[0] * face_img.shape[1] < 100 * 60:
                    upsample = 4
                elif face_img.shape[0] * face_img.shape[1] < 320 * 180:
                    upsample = 3
                elif face_img.shape[0] * face_img.shape[1] < 640 * 360:
                    upsample = 2
                elif face_img.shape[0] * face_img.shape[1] < 1280 * 720:
                    upsample = 1
                face_det = face_detector(face_img, upsample)
                for fd in face_det:
                    d = fd.rect
                    fx1, fy1, fx2, fy2 = d.left(), d.top(), d.right(), d.bottom()
                    mosaic(f0, (x1+fx1, y1+fy1, x1+fx2, y1+fy2), block=mosaic_size)
            # class == Head
            elif int(d[5]) == 1:
                # Process mosaic head box
                x1, y1, x2, y2 = map(int, d[:4])
                if is_show:
                    cv2.rectangle(f1, (x1, y1), (x2, y2), (96, 96, 216), thickness=2,
                                  lineType=cv2.LINE_AA)
                mosaic(f0, (x1, y1, x2, y2), block=mosaic_size)

        # Process mosaic by detecting faces in dlib
        for wd in whole_det:
            wd = wd.rect
            wx1, wy1, wx2, wy2 = wd.left(), wd.top(), wd.right(), wd.bottom()
            if is_show:
                cv2.rectangle(f1, (wx1, wy1), (wx2, wy2), (96, 96, 216), thickness=2,
                              lineType=cv2.LINE_AA)
            mosaic(f0, (wx1, wy1, wx2, wy2), block=5)

        # 그려서 보여주기
        if is_show:
            if f1.shape[0] >= 1080:
                f1 = cv2.resize(f1, (int(f1.shape[1] * 0.8), int(f1.shape[0] * 0.8)))
            if f0.shape[0] >= 1080:
                f0 = cv2.resize(f0, (int(f0.shape[1] * 0.8), int(f0.shape[0] * 0.8)))
            cv2.imshow("orig", f1)
            cv2.imshow("_", f0)
            k = cv2.waitKey(0)
            if k == ord("q"):
                return
            elif k == ord('y'):
                is_show = False
        f_cnt += 1
        if save_dir:
            cv2.imwrite(os.path.join(save_dir, os.path.basename(inp)), f0)
    et = time.time()
    if f_cnt:
        logger.info(f"{f_cnt} images spend {et - st:.4f} sec.")


def args_parse():
    parser = argparse.ArgumentParser(description="if --show, you can click 'y' to image invisible. "
                                                 "Mosaic size is 10 by default.")
    parser.add_argument('-i', '--inputs', required=True,
                        help='image file or directory path')
    parser.add_argument('-s', '--save_path', default="")
    parser.add_argument('--show', action='store_true')
    parser.add_argument('-m', '--mosaic_size', type=int, default=10,
                        help='mosaic block size')
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":

    args = args_parse()
    process_mosaic(args)
