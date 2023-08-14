"""
datavoucher 4 - withyou
human object classes by gender: male, female
manual labeling by click two points and class input by keyboard number
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


img = None
mouseX, mouseY = 0, 0
box_point = []
label_info = []


def draw_event(event, x, y, flags, param):
    global mouseX, mouseY, img, box_point, label_info
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        cv2.circle(img, (x, y), 3, (32, 216, 32), -1)
        box_point.append((x, y))
        if len(box_point) == 2:
            crop_img = img[box_point[0][1]:box_point[1][1], box_point[0][0]:box_point[1][0]]
            cv2.imshow("crop", crop_img)
            key = cv2.waitKey(0)
            _class = 0
            if key == ord("1"):
                _class = 1
            elif key == ord("2"):
                _class = 2
            elif key == ord("d"):
                box_point = []
                cv2.destroyWindow("crop")
                return
            else:
                raise Exception("Wrong Key input")
            label_info.append((box_point[0], box_point[1], _class))
            cv2.rectangle(img, box_point[0], box_point[1], (32, 216, 32), 1, cv2.LINE_AA)
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
    get_logger().info(f"Start dv4 manual annotation script.")
    IMGS_DIR = opt.imgs_dir
    get_logger().info(f"Input Directory is {IMGS_DIR}")

    # detector = ObjectDetector(cfg=cfg)
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
            "categories": [{"id": 0, "name": "background", "supercategory": "background"},
                           {"id": 1, "name": "male", "supercategory": "person"},
                           {"id": 2, "name": "female", "supercategory": "person"}],
            "images": [],
            "annotations": []
        }
        img_ids = 0
        anno_ids = 0

    is_out = False

    image_extension = ['.jpg', '.png', '.jpeg', '.bmp']
    IMGS = [i for i in os.listdir(IMGS_DIR) if os.path.splitext(i)[-1].lower() in image_extension]
    IMGS.sort()
    for idx, i in enumerate(IMGS):
        if is_out is True:
            break

        img_file = os.path.join(IMGS_DIR, i)
        get_logger().info(f"process {img_file}.")
        f = cv2.imread(img_file)
        if os.path.exists(img_file) is True and f is None:      # File 경로에 한글
            f = open(img_file.encode("utf8"), mode="rb")
            bs = bytearray(f.read())
            arr = np.asarray(bs, dtype=np.uint8)
            f = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

        winname = f"{idx + 1}/{len(IMGS)}"
        cv2.namedWindow(winname)
        cv2.setMouseCallback(winname, draw_event, winname)

        img_info = {
            "id": img_ids,
            "license": 1,
            "file_name": os.path.join(opt.type, i),
            "height": f.shape[0],
            "width": f.shape[1],
            "data_captured": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        }

        # image resize
        orig_img_size = (f.shape[0], f.shape[1])
        edit_img_size = orig_img_size
        global img
        img = f
        while f.shape[0] >= 1080:
            f = cv2.resize(f, (int(f.shape[1] * 0.8), int(f.shape[0] * 0.8)))
            img = f
            edit_img_size = (f.shape[0], f.shape[1])

        cv2.imshow(winname, f)
        k = cv2.waitKey(0)
        if k == ord('q'):
            get_logger().info("-- CV2 Stop --")
            break
        elif k == ord(" "):
            global label_info
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
                basic_fmt["annotations"].append(anno_info)
                obj_classes[l_info[2]] += 1

            basic_fmt["images"].append(img_info)
            img_ids += 1
            get_logger().info(
                f"Save label {img_file}. Add {len(label_info)} boxes."
            )

            label_info = []
        cv2.destroyWindow(winname)

        new_path = os.path.join(IMGS_DIR, opt.type, i)
        if not os.path.exists(os.path.dirname(new_path)):
            os.makedirs(os.path.dirname(new_path))
        shutil.move(img_file, new_path)

    with open(os.path.join(opt.json_file), 'w') as outfile:
        json.dump(basic_fmt, outfile, indent=2)
    get_logger().info(f"obj classes: {obj_classes}")


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--imgs_dir', required=True,
                        help='image directory path')
    parser.add_argument('-j', '--json_file', required=True,
                        help="if write this file, append annotations")
    parser.add_argument('-t', '--type', default='train',
                        help='type is in [train, val]. this option write file_path {type}/img_file')
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    args = args_parse()

    update_config(cfg, args='./configs/annotate.yaml')
    init_logger(cfg)

    main(args)
