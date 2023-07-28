import json
import os
import sys
import argparse
from datetime import datetime
from collections import defaultdict
import cv2

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
        img_ids = int(basic_fmt['images'][-1]['id']) + 1 if len(basic_fmt['images']) != 0 else 0
        anno_ids = 0
        if len(basic_fmt['annotations']) != 0:
            anno_ids = int(basic_fmt["annotations"][-1]['id']) + 1
            for anno in basic_fmt['annotations']:
                obj_classes[int(anno['category_id'])] += 1
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
        basic_fmt["images"].append(img_info)

        # bbox annotation
        for d in _det:
            if d[4] < float(opt.confidence):
                continue
            x1, y1, x2, y2 = map(int, d[:4])
            w, h = x2-x1, y2-y1
            fc = f.copy()
            cv2.rectangle(fc, (x1, y1), (x2, y2), (96, 96, 216), thickness=2, lineType=cv2.LINE_AA)

            cv2.imshow(img_file, fc)
            category_id = 0
            k = cv2.waitKey(0)
            if k == ord('q'):
                get_logger().info("-- CV2 Stop --")
                is_out = True
                break
            elif k == ord('1'):
                category_id = 1
            elif k == ord('0'):
                category_id = 0

            anno_info = {"id": anno_ids,
                         "image_id": img_ids,
                         "category_id": category_id,
                         "bbox": [x1, y1, w, h],
                         "area": w*h,
                         "segmentation": [],
                         "iscrowd": 0}
            obj_classes[category_id] += 1
            basic_fmt["annotations"].append(anno_info)
            anno_ids += 1
        img_ids += 1

    with open(os.path.join(opt.json_file), 'w') as outfile:
        json.dump(basic_fmt, outfile, indent=2)


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
