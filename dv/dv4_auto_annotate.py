# -*- coding: utf-8 -*-
"""
datavoucher 4 - withyou
human object classes by gender: male, female
custom trained model detection result
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


def main(opt=None):
    get_logger().info(f"Start dv4 auto annotation script.")
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
    for IMGS_DIR in IMGS_DIRS:
        if is_out is True:
            break

        show_img = True
        IMGS = [i for i in os.listdir(IMGS_DIR) if os.path.splitext(i)[-1].lower() in image_extension]
        IMGS.sort()
        get_logger().info(f"process {IMGS_DIR} ..")
        _imgs = tqdm(IMGS, f'{IMGS_DIR} images ..')
        for i in _imgs:
            if is_out is True:
                break

            img_file = os.path.join(IMGS_DIR, i)
            f0 = cv2.imread(img_file)
            if os.path.exists(img_file) is True and f0 is None:  # File 경로에 한글
                f0 = open(img_file.encode("utf8"), mode="rb")
                bs = bytearray(f0.read())
                arr = np.asarray(bs, dtype=np.uint8)
                f0 = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

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
            im_h, im_w = f0.shape[0], f0.shape[1]
            for d in _det:
                x1, y1, x2, y2 = map(int, d[:4])
                w, h = x2 - x1, y2 - y1

                _cls_idx = int(d[5])
                if detector.names[_cls_idx] == 'female':
                    category_id = 2
                    b_color = (16, 16, 216)
                elif detector.names[_cls_idx] == 'male':
                    category_id = 1
                    b_color = (216, 16, 16)
                else:
                    raise Exception("Wrong Class index")
                if show_img is True:
                    cv2.rectangle(f1, (x1, y1), (x2, y2), b_color, thickness=1, lineType=cv2.LINE_AA)

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

            if show_img is True:
                # image resize
                orig_img_size = (f0.shape[0], f0.shape[1])
                edit_img_size = orig_img_size
                while f1.shape[0] >= 1080:
                    f1 = cv2.resize(f1, (int(f1.shape[1] * 0.8), int(f1.shape[0] * 0.8)))
                    edit_img_size = (f1.shape[0], f1.shape[1])
                cv2.imshow(winname, f1)

                k = cv2.waitKey(0)
                cv2.destroyAllWindows()
                if k == ord('q'):
                    get_logger().info("-- CV2 Stop --")
                    is_out = True
                    break
                elif k == ord('y'):
                    show_img = False
                    basic_fmt["images"].append(img_info)
                    for k, v in c_id.items():
                        obj_classes[k] += v
                    for _anno_info in tmp_annos:
                        basic_fmt["annotations"].append(_anno_info)
                elif k == ord(" "):
                    basic_fmt["images"].append(img_info)
                    for k, v in c_id.items():
                        obj_classes[k] += v
                    for _anno_info in tmp_annos:
                        basic_fmt["annotations"].append(_anno_info)
                else:
                    continue

            new_path = os.path.join(IMGS_DIR, opt.type, i)
            if not os.path.exists(os.path.dirname(new_path)):
                os.makedirs(os.path.dirname(new_path))
            shutil.move(img_file, new_path)
            img_ids += 1

    with open(os.path.join(opt.json_file), 'w') as outfile:
        json.dump(basic_fmt, outfile, indent=2)
    get_logger().info(f"obj classes: {obj_classes}")


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--imgs_dir', required=True, nargs='+',
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
