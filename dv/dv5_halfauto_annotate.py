# -*- coding: utf-8 -*-
"""
datavoucher 5 - quantumai
human group object classes by gender: homo, hetero, unknown
custom trained model detection result
detection based select(include discard) + manual labeling
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
label_info = []


def cvtPoint(x, y, orig_size, cvt_size):
    orig_h, orig_w = orig_size
    cvt_h, cvt_w = cvt_size

    new_x = int(x / orig_w * cvt_w)
    new_y = int(y / orig_h * cvt_h)
    return new_x, new_y


def get_box_point(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    new_pt1 = (min(x1, x2), min(y1, y2))
    new_pt2 = (max(x1, x2), max(y1, y2))
    return new_pt1, new_pt2


def draw_event(event, x, y, flags, param):
    global mouseX, mouseY, img, box_point, label_info
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        cv2.circle(img, (x, y), 3, (32, 216, 32), -1)
        box_point.append((x, y))
        if len(box_point) == 2:
            box_pt1, box_pt2 = get_box_point(box_point[0], box_point[1])
            crop_img = img[box_pt1[1]:box_pt2[1], box_pt1[0]:box_pt2[0]]
            cv2.imshow("crop", crop_img)
            key = cv2.waitKey(0)
            _class = 0
            if key == ord("1"):
                _class = 1
                b_color = (216, 16, 16)  # homo blue
            elif key == ord("2"):
                _class = 2
                b_color = (16, 16, 216)  # hetero red
            elif key == ord("3"):
                _class = 3
                b_color = (16, 216, 16)  # unknown green
            else:
                box_point = []
                cv2.destroyWindow("crop")
                return
            label_info.append((box_pt1, box_pt2, _class))
            cv2.rectangle(img, box_pt1, box_pt2, b_color, 1, cv2.LINE_AA)
            box_point = []
            cv2.destroyWindow("crop")
            print(label_info)
    if event == cv2.EVENT_MOUSEMOVE:
        im = img.copy()
        img_size = im.shape
        cv2.line(im, (x, 0), (x, img_size[0]), (0, 0, 0), 1, cv2.LINE_AA)
        cv2.line(im, (0, y), (img_size[1], y), (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow(param, im)


def main(opt=None):
    get_logger().info(f"Start dv4 halfauto annotation script.")
    IMGS_DIRS = opt.imgs_dir
    get_logger().info(f"Input Directory is {IMGS_DIRS}")

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
                     "description": "datavoucher human gender object detection dataset.",
                     "contributor": "",
                     "url": "",
                     "date_created": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")},
            "licenses": [{"id": 1, "url": "", "name": "Unknown"}],
            "categories": [{"id": 0, "name": "background", "supercategory": "person"},
                           {"id": 1, "name": "homo", "supercategory": "person"},
                           {"id": 2, "name": "hetero", "supercategory": "person"},
                           {"id": 3, "name": "unknown", "supercategory": "person"}],
            "images": [],
            "annotations": []
        }
        img_ids = 0
        anno_ids = 0

    is_out = False
    image_extension = ['.jpg', '.png', '.jpeg', '.bmp']
    for IMGS_DIR in IMGS_DIRS:
        if is_out is True:
            break

        IMGS = [i for i in os.listdir(IMGS_DIR) if os.path.splitext(i)[-1].lower() in image_extension]
        # IMGS.sort()
        get_logger().info(f"process {IMGS_DIR} ..")
        for idx, i in enumerate(IMGS):
            if is_out is True:
                break

            img_file = os.path.join(IMGS_DIR, i)
            get_logger().info(f"process {img_file}.")
            f0 = cv2.imread(img_file)
            if os.path.exists(img_file) is True and f0 is None:  # File 경로에 한글
                f0 = open(img_file.encode("utf8"), mode="rb")
                bs = bytearray(f0.read())
                arr = np.asarray(bs, dtype=np.uint8)
                f0 = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            # Connect click event
            winname = f"{idx + 1}/{len(IMGS)}"
            cv2.namedWindow(winname)
            cv2.setMouseCallback(winname, draw_event, winname)

            # yolov5 human detector
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
            for d in _det:
                x1, y1, x2, y2 = map(int, d[:4])
                _cls_idx = int(d[5])
                if detector.names[_cls_idx] == 'homo':
                    b_color = (255, 128, 128)
                elif detector.names[_cls_idx] == 'hetero':     # male
                    b_color = (128, 128, 255)
                elif detector.names[_cls_idx] == 'unknown':
                    b_color = (128, 255, 128)
                else:
                    raise Exception("Wrong Class index")
                cv2.rectangle(f1, (x1, y1), (x2, y2), b_color, thickness=1, lineType=cv2.LINE_AA)

            # image resize
            orig_img_size = (f0.shape[0], f0.shape[1])
            edit_img_size = orig_img_size
            global img, label_info
            img = f1
            while f1.shape[0] >= 1080:
                f1 = cv2.resize(f1, (int(f1.shape[1] * 0.8), int(f1.shape[0] * 0.8)))
                img = f1
                edit_img_size = (f1.shape[0], f1.shape[1])
            cv2.imshow(winname, f1)

            auto_box = 0
            for _idx, d in enumerate(_det):
                x1, y1, x2, y2 = map(int, d[:4])
                w, h = x2 - x1, y2 - y1
                _cls_idx = int(d[5])
                crop_image = f0[y1:y2, x1:x2]
                b_name = f"{_idx+1}/{len(_det)}"
                cv2.imshow(b_name, crop_image)

                nx1, ny1 = cvtPoint(x1, y1, orig_img_size, edit_img_size)
                nx2, ny2 = cvtPoint(x2, y2, orig_img_size, edit_img_size)

                _k = cv2.waitKey(0)
                if _k == ord('q'):
                    get_logger().info("-- CV2 Stop --")
                    is_out = True
                    cv2.destroyWindow(b_name)
                    break
                elif _k == ord('1'):
                    category_id = 1
                    b_color = (216, 16, 16)  # homo blue
                elif _k == ord('2'):
                    category_id = 2
                    b_color = (16, 16, 216)  # hetero red
                elif _k == ord('3'):
                    category_id = 3
                    b_color = (16, 216, 16)  # homo blue
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
            if k == ord('q'):
                get_logger().info("-- CV2 Stop --")
                is_out = True
                break
            elif k == ord(" "):
                for l_info in label_info:
                    pt1, pt2 = l_info[0], l_info[1]
                    rel_pt1 = (pt1[0] / edit_img_size[1], pt1[1] / edit_img_size[0])
                    rel_pt2 = (pt2[0] / edit_img_size[1], pt2[1] / edit_img_size[0])
                    orig_pt1 = (int(rel_pt1[0] * orig_img_size[1]), int(rel_pt1[1] * orig_img_size[0]))
                    orig_pt2 = (int(rel_pt2[0] * orig_img_size[1]), int(rel_pt2[1] * orig_img_size[0]))
                    w = int(orig_pt2[0] - orig_pt1[0])
                    h = int(orig_pt2[1] - orig_pt1[1])

                    anno_info = {
                        "id": anno_ids,
                        "image_id": img_ids,
                        "category_id": l_info[2],
                        "bbox": [orig_pt1[0], orig_pt1[1], w, h],
                        "area": w * h,
                        "segmentation": [],
                        "iscrowd": 0
                    }
                    anno_ids += 1
                    tmp_annos.append(anno_info)
                    c_id[l_info[2]] += 1

                if is_out is False:
                    basic_fmt["images"].append(img_info)
                    for k, v in c_id.items():
                        obj_classes[k] += v
                    for _anno_info in tmp_annos:
                        basic_fmt["annotations"].append(_anno_info)
                    get_logger().info(
                        f"Save label {img_file}. Add {len(label_info) + auto_box} boxes."
                    )
                label_info = []

                new_path = os.path.join(IMGS_DIR, opt.type, i)
                if not os.path.exists(os.path.dirname(new_path)):
                    os.makedirs(os.path.dirname(new_path))
                shutil.move(img_file, new_path)
                img_ids += 1
            else:
                label_info = []
                continue

    with open(opt.json_file, 'w') as outfile:
        json.dump(basic_fmt, outfile, indent=2, ensure_ascii=False)

    get_logger().info(f"obj classes: {obj_classes}")


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--imgs_dir', required=True, nargs='+',
                        help='image directory path')
    parser.add_argument('-j', '--json_file', required=True,
                        help="if write this file, append annotations")
    parser.add_argument('-t', '--type', default='train',
                        help='type is in [train, val]. this option write file_path {type}/img_file')
    parser.add_argument('-c', '--config', default='./configs/dv5_annotate.yaml',
                        help="annotation configuration yaml file path")
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    args = args_parse()

    update_config(cfg, args=args.config)
    init_logger(cfg)

    main(args)
