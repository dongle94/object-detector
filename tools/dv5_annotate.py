"""
datavoucher 5 - quamtomai
human group object classes by gender: homo, hetero, unknown
"""
import json
import os
import sys
import argparse
import numpy as np
from datetime import datetime
from collections import defaultdict
import cv2
import shutil

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


def get_l2_norm(boxes):
    _det = []
    for b in boxes:
        x1, y1, x2, y2 = map(int, b[:4])
        w, h = x2-x1, y2-y1
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        d = ((2 * 3.14 * 180) / (w + h * 360) * 1000 + 3)
        _det.append(np.asarray((cx, cy, d)))
    arr = np.zeros((len(boxes), len(boxes)), dtype=np.float32)

    for i, d1 in enumerate(_det[:]):
        for j, d2 in enumerate(_det[:]):
            arr[i, j] = np.sqrt(np.sum(np.power(d1-d2, 2)))
    return arr


def get_closed_boxes(arr, threshold=100):
    length = arr.shape[0]
    ret = []
    for i in range(length):
        for j in range(i+1, length, 1):
            if arr[i, j] < threshold:
                ret.append((i, j))
    return ret


def get_merge_boxes(boxes, box_pair):
    new_boxes = []
    for p in box_pair:
        ax1, ay1, ax2, ay2 = map(int, boxes[p[0]][:4])
        bx1, by1, bx2, by2 = map(int, boxes[p[1]][:4])
        x1 = min(ax1, bx1)
        y1 = min(ay1, by1)
        x2 = max(ax2, bx2)
        y2 = max(ay2, by2)
        new_boxes.append((x1, y1, x2, y2))
    return new_boxes


def main(opt=None):
    IMGS_DIR = opt.imgs_dir
    get_logger().info(f"Input Directory is {IMGS_DIR}")

    detector = ObjectDetector(cfg=cfg)
    obj_classes = defaultdict(int)

    if os.path.exists(opt.json_file):
        with open(opt.json_file, 'r') as file:
            basic_fmt = json.load(file)
        get_logger().info(f"{opt.json_file} is exist. append annotation file.")
        img_ids = 0
        if len(basic_fmt['images']) != 0:
            img_ids = int(basic_fmt['images'][-1]['id']) + 1
            get_logger().info(f"last image file name: {basic_fmt['images'][-1]['file_name']}")
        anno_ids = 0
        if len(basic_fmt['annotations']) != 0:
            anno_ids = int(basic_fmt["annotations"][-1]['id']) + 1
            for anno in basic_fmt['annotations']:
                obj_classes[int(anno['category_id'])] += 1
            get_logger().info(f"old object classes: {obj_classes}")
    else:
        get_logger().info(f"{opt.json_file} is not exist. Create new annotation file")
        basic_fmt = {
            "info": {"year": "2023", "version": "1",
                     "description": "",
                     "contributor": "",
                     "url": "",
                     "date_created": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")},
            "licenses": [{"id": 1, "url": "", "name": "Unknown"}],
            "categories": [{"id": 0, "name": "male", "supercategory": "person"},
                           {"id": 1, "name": "female", "supercategory": "person"}],
            "images": [],
            "annotations": []
        }
        img_ids = 0
        anno_ids = 0

    is_out = False

    IMGS = os.listdir(IMGS_DIR)
    IMGS.sort()
    for i in IMGS:
        if is_out is True:
            break
        if os.path.splitext(i)[-1].lower() not in ['.jpg', '.png', '.jpeg', '.bmp']:
            continue

        img_file = os.path.join(IMGS_DIR, i)
        get_logger().info(f"process {img_file}.")
        f0 = cv2.imread(img_file)

        # yolov5 human detector
        im = detector.preprocess(f0)
        _pred = detector.detect(im)
        _pred, _det = detector.postprocess(_pred)

        img_info = {
            "id": img_ids,
            "license": 1,
            "file_name": i,
            "height": f0.shape[0],
            "width": f0.shape[1],
            "data_captured": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        }

        # bbox annotation
        _det = [d for d in _det if d[4] > float(opt.confidence)]
        _arr = get_l2_norm(_det)
        _boxes = get_closed_boxes(_arr, threshold=120)
        print(_boxes)
        _new_boxes = get_merge_boxes(_det, _boxes)

        f1 = f0.copy()
        for i, d in enumerate(_det):
            x1, y1, x2, y2 = map(int, d[:4])
            w, h = x2-x1, y2-y1

            cv2.rectangle(f1, (x1, y1), (x2, y2), (16, 16, 255), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(f1, f"{i}", (x1+10, y1+20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(255, 16, 16), thickness=2)

            d = ((2 * 3.14 * 180) / (w + h * 360) * 1000 + 3)
            cv2.putText(f1, f"d: {d:.3f}", (int(x1+w/2), int(y1+h/2)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(255, 16, 16), thickness=2)

        # group box draw
        for d in _new_boxes:
            f2 = f1.copy()
            x1, y1, x2, y2 = d
            cv2.rectangle(f2, (x1, y1), (x2, y2), (16, 255, 16), thickness=3, lineType=cv2.LINE_AA)

            if f2.shape[0] > 1000:
                f2 = cv2.resize(f2, (int(f0.shape[1] * 0.8), int(f0.shape[0] * 0.8)))
            cv2.imshow(img_file, f2)

            new_frame = f0[y1:y2, x1:x2]
            cv2.imshow("crop", new_frame)

            k = cv2.waitKey(0)
            if k == ord('q'):
                get_logger().info("-- CV2 Stop --")
                is_out = True
                break
            # category_id = 0

            # elif k == ord('1'):
            #     category_id = 1
            # elif k == ord('0'):
            #     category_id = 0
            # elif k == ord('d'):
            #     continue
            #
            # anno_info = {"id": anno_ids,
            #              "image_id": img_ids,
            #              "category_id": category_id,
            #              "bbox": [x1, y1, w, h],
            #              "area": w*h,
            #              "segmentation": [],
            #              "iscrowd": 0}
            # obj_classes[category_id] += 1
            # basic_fmt["annotations"].append(anno_info)
            # anno_ids += 1

        cv2.destroyAllWindows()
        # if is_out is False:
        #     basic_fmt["images"].append(img_info)
        #     new_path = os.path.join(IMGS_DIR, 'already', i)
        #     if not os.path.exists(os.path.dirname(new_path)):
        #         os.makedirs(os.path.dirname(new_path))
        #     shutil.move(img_file, new_path)
        # img_ids += 1

    # with open(os.path.join(opt.json_file), 'w') as outfile:
    #     json.dump(basic_fmt, outfile, indent=2)
    # get_logger().info(f"obj classes: {obj_classes}")


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--imgs_dir', required=True,
                        help='image directory path')
    parser.add_argument('-j', '--json_file', required=True,
                        help="if write this file, append annotations")
    parser.add_argument('-c', '--confidence', default=0.3,
                        help="obj_detector confidence")
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    args = args_parse()

    update_config(cfg, args='./configs/annotate.yaml')
    init_logger(cfg)

    main(args)
