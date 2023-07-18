import os
import cv2
import dlib
import argparse
import time

import numpy as np
from tqdm import tqdm


def mosaic(img: np.ndarray, coord, block=10):

    x1, y1, x2, y2 = coord
    w, h = x2 - x1+1, y2 - y1+1
    if w < block:
        block /= 2
    gap_w, gap_h = int(w/block), int(h/block)
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

    detector = dlib.cnn_face_detection_model_v1('./weights/mmod_human_face_detector.dat')

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
    else:
        print("There is no image files")
        return

    # 이미지 읽으면서
    f_cnt = 0
    st = time.time()
    for inp in tqdm(inps, "Processing image mosaic"):
        img = dlib.load_rgb_image(inp)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # 얼굴인식
        upsample = 0
        if img.shape[0] * img.shape[1] < 640 * 480:
            upsample = 2
        elif img.shape[0] * img.shape[1] < 1280 * 720:
            upsample = 1
        det = detector(img, upsample)
        for d in det:
            d = d.rect
            x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()

            # 얼굴별 모자이크 처리
            mosaic(img, (x1, y1, x2, y2), block=8)

        # 그려서 보여주기
        if is_show:
            cv2.imshow("_", img)
            cv2.waitKey(0)
        f_cnt += 1
        if save_dir:
            cv2.imwrite(os.path.join(save_dir, os.path.basename(inp)), img)
    et = time.time()
    if f_cnt:
        print(f"{f_cnt} images spend {(et - st) / f_cnt:.4f} sec.")


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
