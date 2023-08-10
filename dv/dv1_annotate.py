# -*- coding: utf-8 -*-
"""
datavoucher 1 - 다올씨앤디
Product object 17 classes
Manual labeling script
"""
import json
import os
import sys
import argparse
import cv2
from datetime import datetime
from collections import defaultdict
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


PROD_CLASS = {
    "background": 0,
    "닭윙1.3kg": 1,
    "시즈닝100g": 2,
    "시즈닝500g": 3,
    "닭볶음용1.3kg": 4,
    "닭다리살1kg": 5,
    "닭갈비간장500g": 6,
    "닭볶음탕용600g": 7,
    "허니버터소스1kg": 8,
    "소떡소떡소스1kg": 9,
    "닭갈비고추장500g": 10,
    "갈릭디핑소스500g": 11,
    "치즈치폴레소스500g": 12,
    "13cls": 13,
    "14cls": 14,
    "15cls": 15,
    "16cls": 16,
    "17cls": 17,
}


img = None
mouseX, mouseY = 0, 0
box_point = []


def draw_event(event, x, y, flags, param):
    global mouseX, mouseY, img, box_point
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        cv2.circle(img, (x, y), 5, (32, 216, 32), -1)
        box_point.append((x, y))
        if len(box_point) == 2:
            cv2.rectangle(img, box_point[0], box_point[1], (32, 32, 216), 2, cv2.LINE_AA)
        cv2.imshow(param, img)
    if event == cv2.EVENT_MOUSEMOVE:
        im = img.copy()
        img_size = im.shape
        cv2.line(im, (x, 0), (x, img_size[0]), (0, 0, 0), 2, cv2.LINE_AA)
        cv2.line(im, (0, y), (img_size[1], y), (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow(param, im)

def main(opt=None):
    get_logger().info(f"Start dv1 annotation script. Object class is {opt.class_num}")
    IMGS_DIR = opt.imgs_dir
    get_logger().info(f"Input Directory is {IMGS_DIR}")

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
                     "description": "datavoucher dataset",
                     "contributor": "",
                     "url": "",
                     "date_created": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")},
            "licenses": [{"id": 1, "url": "", "name": "Unknown"}],
            "categories": [
                {"id": 0, "name": "background", "supercategory": "background"},
                {"id": 1, "name": "닭윙1.3kg", "supercategory": "product"},
                {"id": 2, "name": "시즈닝100g", "supercategory": "product"},
                {"id": 3, "name": "시즈닝500g", "supercategory": "product"},
                {"id": 4, "name": "닭볶음용1.3kg", "supercategory": "product"},
                {"id": 5, "name": "닭다리살1kg", "supercategory": "product"},
                {"id": 6, "name": "닭갈비간장500g", "supercategory": "product"},
                {"id": 7, "name": "닭볶음탕용600g", "supercategory": "product"},
                {"id": 8, "name": "허니버터소스1kg", "supercategory": "product"},
                {"id": 9, "name": "소떡소떡소스1kg", "supercategory": "product"},
                {"id": 10, "name": "닭갈비고추장500g", "supercategory": "product"},
                {"id": 11, "name": "갈릭디핑소스500g", "supercategory": "product"},
                {"id": 12, "name": "치즈치폴레소스500g", "supercategory": "product"},
                {"id": 13, "name": "14cls", "supercategory": "product"},
                {"id": 14, "name": "15cls", "supercategory": "product"},
                {"id": 15, "name": "16cls", "supercategory": "product"},
                {"id": 16, "name": "17cls", "supercategory": "product"},
                {"id": 17, "name": "18cls", "supercategory": "product"},
            ],
            "images": [],
            "annotations": []
        }
        img_ids = 0
        anno_ids = 0

    image_extension = ['.jpg', '.png', '.jpeg', '.bmp']
    IMGS = [i for i in os.listdir(IMGS_DIR) if os.path.splitext(i)[-1].lower() in image_extension]
    IMGS.sort()

    for idx, i in enumerate(IMGS):

        img_file = os.path.join(IMGS_DIR, i)
        get_logger().info(f"process {img_file}.")
        f = cv2.imread(img_file)

        # Connect click event
        winname = f"{idx+1}/{len(IMGS)}"
        cv2.namedWindow(winname)
        cv2.setMouseCallback(winname, draw_event, winname)

        img_info = {
            "id": img_ids,
            "license": 1,
            "file_name": i,
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
            f = cv2.resize(f, (int(f.shape[1] / 2), int(f.shape[0] / 2)))
            img = f
            edit_img_size = (f.shape[0], f.shape[1])

        # winname에 한글 입력 불가
        cv2.imshow(winname, f)
        k = cv2.waitKey(0)
        if k == ord('q'):
            get_logger().info("-- CV2 Stop --")
            break
        elif k == ord(" "):
            global box_point
            if len(box_point) == 2:
                pt1, pt2 = box_point[0], box_point[1]
                rel_pt1 = (pt1[0]/edit_img_size[1], pt1[1]/edit_img_size[0])
                rel_pt2 = (pt2[0]/edit_img_size[1], pt2[1]/edit_img_size[1])
                orig_pt1 = (int(rel_pt1[0] * orig_img_size[1]), int(rel_pt1[1] * orig_img_size[0]))
                orig_pt2 = (int(rel_pt2[0] * orig_img_size[1]), int(rel_pt2[1] * orig_img_size[0]))
                w = int(orig_pt2[0] - orig_pt1[0])
                h = int(orig_pt2[1] - orig_pt1[1])
                box_point = []
            else:
                raise Exception("2 points not clicked!")
            anno_info = {
                "id": anno_ids,
                "image_id": img_ids,
                "category_id": opt.class_num,
                "bbox": [orig_pt1[0], orig_pt1[1], w, h],
                "area": w*h,
                "segmentation": [],
                "iscrowd": 0
            }
            get_logger().info(
                f"Save label {img_file}. Box point is {orig_pt1, w, h}. resize box point is {pt1, pt2}."
                f"relative point is ({rel_pt1[0]:.4f}, {rel_pt1[1]:.4f}) ({rel_pt2[0]:.4f}, {rel_pt2[1]:.4f})"
            )

            # add annotation
            basic_fmt["images"].append(img_info)
            basic_fmt["annotations"].append(anno_info)
            img_ids += 1
            anno_ids += 1
            obj_classes[opt.class_num] += 1

            new_path = os.path.join(IMGS_DIR, 'already', i)
            if not os.path.exists(os.path.dirname(new_path)):
                os.makedirs(os.path.dirname(new_path))
            shutil.move(img_file, new_path)

        cv2.destroyAllWindows()

    with open(os.path.join(opt.json_file), 'w') as outfile:
        json.dump(basic_fmt, outfile, indent=1)
    get_logger().info(f"Stop Annotation. Obj classes: {obj_classes}")


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--imgs_dir', required=True,
                        help='image directory path')
    parser.add_argument('-j', '--json_file', required=True,
                        help="if write this file, append annotations")
    parser.add_argument('-c', '--class_num', required=True, type=int,
                        help="object class number 1~17")
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    args = args_parse()

    update_config(cfg, args='./configs/annotate.yaml')
    init_logger(cfg)

    main(args)
