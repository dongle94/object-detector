# -*- coding: utf-8 -*-
"""
datavoucher p2 - 와이이노베이션
object 3 classes
Manual labeling script
"""

import json
import os
import sys
import argparse
from datetime import datetime
from collections import defaultdict
import cv2
import shutil
import numpy as np

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


img = None
mouseX, mouseY = 0, 0
box_point = []


def cvtPoint(x, y, orig_size, cvt_size):
    orig_h, orig_w = orig_size
    cvt_h, cvt_w = cvt_size

    new_x = int(x / orig_w * cvt_w)
    new_y = int(y / orig_h * cvt_h)
    return new_x, new_y


def get_box_point(pt1, pt2):
    """
    return box point xyxy with 2 points
    :param pt1:
    :param pt2:
    :return new_pt1, new_pt2:
    """
    x1, y1 = pt1
    x2, y2 = pt2
    new_pt1 = (min(x1, x2), min(y1, y2))
    new_pt2 = (max(x1, x2), max(y1, y2))
    return new_pt1, new_pt2


def draw_event(event, x, y, flags, param):
    global mouseX, mouseY, img, box_point
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        cv2.circle(img, (x, y), 5, (32, 216, 32), -1)
        box_point.append((x, y))
        if len(box_point) % 2 == 0:
            box_pt1, box_pt2 = get_box_point(box_point[-2], box_point[-1])
            cv2.rectangle(img, box_pt1, box_pt2, (32, 32, 216), 2, cv2.LINE_AA)
        cv2.imshow(param, img)
    if event == cv2.EVENT_MOUSEMOVE:
        im = img.copy()
        img_size = im.shape
        cv2.line(im, (x, 0), (x, img_size[0]), (0, 0, 0), 2, cv2.LINE_AA)
        cv2.line(im, (0, y), (img_size[1], y), (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow(param, im)


def main(opt=None):
    get_logger().info(f"Start dv p1 annotation script. Object class is {opt.class_num}")
    IMGS_DIR = opt.imgs_dir
    get_logger().info(f"Input Directory is {IMGS_DIR}")

    detector = ObjectDetector(cfg=cfg) if cfg.DET_MODEL_PATH != "" else None

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
                     "description": "datavoucher dangerous object detection dataset",
                     "contributor": "",
                     "url": "",
                     "date_created": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")},
            "licenses": [{"id": 1, "url": "", "name": "Unknown"}],
            "categories": [
                {"id": 0, "name": "background", "supercategory": "background"},
                {"id": 1, "name": "knife", "supercategory": "dangerous"},
                {"id": 2, "name": "cigarette", "supercategory": "dangerous"},
                #{"id": 3, "name": "weapon", "supercategory": "dangerous"}
                {"id": 3, "name": "gun", "supercategory": "dangerous"},
                {"id": 4, "name": "axe", "supercategory": "dangerous"}
            ],
            "images": [],
            "annotations": []
        }
        img_ids = 0
        anno_ids = 0

    image_extension = ['.jpg', '.png', '.jpeg', '.bmp']

    IMGS = [i for i in os.listdir(IMGS_DIR) if os.path.splitext(i)[-1].lower() in image_extension]
    # IMGS.sort()

    is_out = False
    for idx, i in enumerate(IMGS):
        if is_out is True:
            break

        img_file = os.path.join(IMGS_DIR, i)
        get_logger().info(f"process {img_file}.")
        f0 = cv2.imread(img_file)
        if os.path.exists(img_file) is True and f0 is None:      # File 경로에 한글
            f0 = open(img_file.encode("utf8"), mode="rb")
            bs = bytearray(f0.read())
            arr = np.asarray(bs, dtype=np.uint8)
            f0 = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

        # Connect click event
        winname = f"{idx+1}/{len(IMGS)}"
        cv2.namedWindow(winname)
        cv2.setMouseCallback(winname, draw_event, winname)

        # yolov5 human detector
        if detector is not None:
            im = detector.preprocess(f0)
            _pred = detector.detect(im)
            _pred, _det = detector.postprocess(_pred)

        img_info = {
            "id": img_ids,
            "license": 1,
            "file_name": os.path.join(opt.type, i),
            "height": f0.shape[0],
            "width": f0.shape[1],
            "data_captured": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        }

        # bbox annotation
        c_id = defaultdict(int)
        tmp_annos = []

        f1 = f0.copy()
        # Draw boxes
        if detector is not None:
            for d in _det:
                x1, y1, x2, y2 = map(int, d[:4])
                _cls_idx = int(d[5])
                if detector.names[_cls_idx] == 'knife':
                    b_color = (216, 48, 216)
                elif detector.names[_cls_idx] == 'cigarette':
                    b_color = (48, 48, 216)
                elif detector.names[_cls_idx] == 'axe':
                    b_color = (216, 48, 48)
                elif detector.names[_cls_idx] == 'gun':\
                    b_color = (48, 216, 48)
                else:
                    raise Exception(f"Wrong Class index: {_cls_idx}")
                cv2.rectangle(f1, (x1, y1), (x2, y2), b_color, thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(f1,
                            f"Class: {_cls_idx, detector.names[_cls_idx]}",
                            (x1+4, y1+25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255), 1)

        # image resize
        orig_img_size = (f0.shape[0], f0.shape[1])
        edit_img_size = orig_img_size
        global img
        img = f1
        while f1.shape[0] >= 1080:
            f1 = cv2.resize(f1, (int(f1.shape[1] * 0.8), int(f1.shape[0] * 0.8)))
            img = f1
            edit_img_size = (f1.shape[0], f1.shape[1])
        cv2.imshow(winname, f1)

        auto_box = 0
        # auto labeling
        if detector is not None:
            for _idx, d in enumerate(_det):
                x1, y1, x2, y2 = map(int, d[:4])
                w, h = x2 - x1, y2 - y1
                _cls_idx = int(d[5])
                crop_image = f0[y1:y2, x1:x2]
                while crop_image.shape[0] >= 1080:
                    crop_image = cv2.resize(crop_image,
                                            (int(crop_image.shape[1] * 0.5), int(crop_image.shape[0] * 0.5)))
                b_name = f"{_idx + 1}/{len(_det)}"
                cv2.imshow(b_name, crop_image)

                nx1, ny1 = cvtPoint(x1, y1, orig_img_size, edit_img_size)
                nx2, ny2 = cvtPoint(x2, y2, orig_img_size, edit_img_size)

                _k = cv2.waitKey(0)
                if _k == ord('q'):
                    get_logger().info("-- CV2 Stop --")
                    is_out = True
                    cv2.destroyWindow(b_name)
                    break
                elif _k == ord(' '):
                    category_id = _cls_idx
                    b_color = (48, 216, 216)
                else:
                    cv2.destroyWindow(b_name)
                    continue
                auto_box += 1
                cv2.rectangle(f1, (nx1, ny1), (nx2, ny2), b_color, thickness=1, lineType=cv2.LINE_AA)
                cv2.imshow(winname, f1)
                cv2.destroyWindow(b_name)

                anno_info = {"id": anno_ids,
                             "image_id": img_ids,
                             "category_id": category_id,
                             "bbox": [x1, y1, w, h],
                             "area": w * h,
                             "segmentation": [],
                             "iscrowd": 0}
                tmp_annos.append(anno_info)
                anno_ids += 1
                c_id[category_id] += 1

        k = cv2.waitKey(0)
        cv2.destroyAllWindows()
        global box_point
        if k == ord('q'):
            get_logger().info("-- CV2 Stop --")
            is_out = True
            break
        elif k == ord(" "):
            if len(box_point) % 2 == 0:
                for box_i in range(0, len(box_point), 2):
                    pt1, pt2 = get_box_point(box_point[box_i], box_point[box_i+1])
                    rel_pt1 = (pt1[0]/edit_img_size[1], pt1[1]/edit_img_size[0])
                    rel_pt2 = (pt2[0]/edit_img_size[1], pt2[1]/edit_img_size[0])
                    orig_pt1 = (int(rel_pt1[0] * orig_img_size[1]), int(rel_pt1[1] * orig_img_size[0]))
                    orig_pt2 = (int(rel_pt2[0] * orig_img_size[1]), int(rel_pt2[1] * orig_img_size[0]))
                    w = int(orig_pt2[0] - orig_pt1[0])
                    h = int(orig_pt2[1] - orig_pt1[1])

                    anno_info = {
                        "id": anno_ids,
                        "image_id": img_ids,
                        "category_id": opt.class_num,
                        "bbox": [orig_pt1[0], orig_pt1[1], w, h],
                        "area": w * h,
                        "segmentation": [],
                        "iscrowd": 0
                    }
                    anno_ids += 1
                    tmp_annos.append(anno_info)
                    c_id[opt.class_num] += 1

                # add annotation
                if is_out is False:
                    basic_fmt["images"].append(img_info)
                    for k, v in c_id.items():
                        obj_classes[k] += v
                    for _anno_info in tmp_annos:
                        basic_fmt["annotations"].append(_anno_info)
                    get_logger().info(
                        f"Save label {img_file}. Add {len(box_point)/2 + auto_box} boxes."
                    )
                box_point = []

                new_path = os.path.join(IMGS_DIR, opt.type, i)
                if not os.path.exists(os.path.dirname(new_path)):
                    os.makedirs(os.path.dirname(new_path))
                shutil.move(img_file, new_path)
                img_ids += 1
            else:
                box_point = []
                print("2 points not clicked!")
                break
        else:
            box_point = []
            get_logger().info(f"Pass image {img_file}.")
            cv2.destroyAllWindows()
            continue


    with open(os.path.join(opt.json_file), 'w') as outfile:
        json.dump(basic_fmt, outfile, indent=1, ensure_ascii=False)
    get_logger().info(f"Stop Annotation. Obj classes: {obj_classes}")


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--imgs_dir', required=True,
                        help='image directory path')
    parser.add_argument('-j', '--json_file', required=True,
                        help="if write this file, append annotations")
    parser.add_argument('-t', '--type', default='train',
                        help='type is in [train, val]. this option write file_path {type}/img_file')
    parser.add_argument('-cn', '--class_num', required=True, type=int, choices=[1, 2, 3, 4],
                        help="object class number 1~3")
    parser.add_argument('-c', '--config', default='./configs/dvp2.yaml',
                        help="annotate.yaml config file path")
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    args = args_parse()

    update_config(cfg, args=args.config)
    init_logger(cfg)

    main(args)
