"""
datavoucher 4 - withyou
human object classes by gender: male, female
yolov5 x6 model + homogenus(gender classification) auto labeling version
"""
import json
import os
import sys
import argparse
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
        f = cv2.imread(img_file)

        # yolov5 human detector
        im = detector.preprocess(f)
        _pred = detector.detect(im)
        _pred, _det = detector.postprocess(_pred)

        img_info = {
            "id": img_ids,
            "license": 1,
            "file_name": i,
            "height": f.shape[0],
            "width": f.shape[1],
            "data_captured": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        }

        # bbox annotation
        c_id = defaultdict(int)
        tmp_annos = []
        for d in _det:
            if d[4] < float(opt.confidence):
                continue
            x1, y1, x2, y2 = map(int, d[:4])
            w, h = x2-x1, y2-y1
            fc = f.copy()
            cv2.rectangle(fc, (x1, y1), (x2, y2), (16, 16, 255), thickness=2, lineType=cv2.LINE_AA)

            if fc.shape[0] > 1000:
                fc = cv2.resize(fc, (int(fc.shape[1] * 0.8), int(fc.shape[0] * 0.8)))
            cv2.imshow(img_file, fc)

            category_id = 0
            k = cv2.waitKey(0)
            if k == ord('q'):
                get_logger().info("-- CV2 Stop --")
                is_out = True
                break
            elif k == ord('1'):
                category_id = 0
            elif k == ord('2'):
                category_id = 1
            elif k == ord('d'):
                continue

            anno_info = {"id": anno_ids,
                         "image_id": img_ids,
                         "category_id": category_id,
                         "bbox": [x1, y1, w, h],
                         "area": w*h,
                         "segmentation": [],
                         "iscrowd": 0}
            tmp_annos.append(anno_info)
            anno_ids += 1
            c_id[category_id] += 1

        cv2.destroyAllWindows()
        if is_out is False:
            for k, v in c_id.items():
                obj_classes[k] += v
            for _anno_info in tmp_annos:
                basic_fmt["annotations"].append(_anno_info)

            basic_fmt["images"].append(img_info)
            new_path = os.path.join(IMGS_DIR, 'already', i)
            if not os.path.exists(os.path.dirname(new_path)):
                os.makedirs(os.path.dirname(new_path))
            shutil.move(img_file, new_path)
        img_ids += 1

    with open(os.path.join(opt.json_file), 'w') as outfile:
        json.dump(basic_fmt, outfile, indent=2)
    get_logger().info(f"obj classes: {obj_classes}")


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
