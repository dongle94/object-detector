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

    update_config(cfg, args='./configs/mosaic.yaml')
    init_logger(cfg)

    detector = ObjectDetector(cfg=cfg)
    face_detector = dlib.cnn_face_detection_model_v1('./weights/mmod_human_face_detector.dat')

    # 디렉토리 읽기
    is_file, is_dir = os.path.isfile(inps), os.path.isdir(inps)
    print(f"{is_file=}, {is_dir=}")
    if save_dir and not os.path.exists(save_dir):
        print(f"make dir {save_dir}")
        os.makedirs(save_dir)

    # 이미지 리스트 가공
    if is_file and isinstance(inps, str):
        inps = [os.path.abspath(inps)]
    elif is_dir:
        files = os.listdir(inps)
        inps = [os.path.abspath(os.path.join(inps, f)) for f in files
                if os.path.isfile(os.path.abspath(os.path.join(inps, f)))]
        inps.sort()
    else:
        print("There is no image files")
        return

    # 이미지 loop start
    f_cnt = 0
    st = time.time()
    for inp in tqdm(inps, "Processing image mosaic"):
        img = cv2.imread(inp)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        im = detector.preprocess(img)
        _pred = detector.detect(im)
        _pred, _det = detector.postprocess(_pred)

        orig_img = img.copy()

        # whole image 1st process
        upsample = 0
        if img.shape[0] * img.shape[1] < 640 * 480:
            upsample = 2
        elif img.shape[0] * img.shape[1] < 1280 * 720:
            upsample = 1
        whole_det = face_detector(img, upsample)

        # human boxes 2nd process
        for d in _det:
            # threshold filtering
            if d[4] < 0.3:
                continue
            if d[5] == 0:
                x1, y1, x2, y2 = map(int, d[:4])
                w, h = x2-x1, y2-y1

                face_img = img[y1:y2, x1:x2]
                # 사람객체 별 얼굴인식 시도 후 모자이크 처리
                upsample = 0
                if face_img.shape[0] * face_img.shape[1] < 100 * 60:
                    upsample = 5
                if face_img.shape[0] * face_img.shape[1] < 320 * 180:
                    upsample = 4
                elif face_img.shape[0] * face_img.shape[1] < 640 * 360:
                    upsample = 3
                elif face_img.shape[0] * face_img.shape[1] < 1280 * 720:
                    upsample = 2
                face_det = face_detector(face_img, upsample)
                for fd in face_det:
                    d = fd.rect
                    fx1, fy1, fx2, fy2 = d.left(), d.top(), d.right(), d.bottom()
                    mosaic(img, (x1+fx1, y1+fy1, x1+fx2, y1+fy2), block=8)
            elif d[5] == 1:
                # 머리 박스 모자이크 처리
                x1, y1, x2, y2 = map(int, d[:4])
                mosaic(img, (x1, y1, x2, y2), block=4)

        # 전체 이미지에 대해서 인식되는 얼굴 모자이크 처리
        for wd in whole_det:
            wd = wd.rect
            wx1, wy1, wx2, wy2 = wd.left(), wd.top(), wd.right(), wd.bottom()
            cv2.rectangle(orig_img, (wx1, wy1), (wx2, wy2), (96, 96, 216), thickness=2,
                          lineType=cv2.LINE_AA)
            mosaic(img, (wx1, wy1, wx2, wy2), block=8)

        # 시간 부분 마스킹
        mosaic(img, (1450, 50, 1850, 100), block=10)
        #cv2.rectangle(img, (1450, 50), (1850, 90), (216, 216, 216), -1)


        # 그려서 보여주기
        if is_show:
            if orig_img.shape[1] * orig_img.shape[0] > 1280 * 720:
                orig_img = cv2.resize(orig_img, (int(orig_img.shape[1]/2), int(orig_img.shape[0]/2)))
            if img.shape[1] * img.shape[0] > 1280 * 720:
                img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))
            cv2.imshow("orig", orig_img)
            cv2.imshow("_", img)
            if cv2.waitKey(0) == ord("q"):
                return
        f_cnt += 1
        if save_dir:
            cv2.imwrite(os.path.join(save_dir, os.path.basename(inp)), img)
    et = time.time()
    if f_cnt:
        print(f"{f_cnt} images spend {et - st:.4f} sec.")


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputs', required=True,
                        help='image file or directory path')
    parser.add_argument('-s', '--save_path', default="")
    parser.add_argument('--show', action='store_true')
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    args = args_parse()
    process_mosaic(args)
