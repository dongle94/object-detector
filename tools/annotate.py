import json
import os
import sys
from datetime import datetime
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

def main():
    IMGS_DIR = '/home/dongle94/Videos/datavoucher/quantom/20210505200137_DDOBONG_ch2'
    update_config(cfg, args='./configs/annotate.yaml')
    init_logger(cfg)
    detector = ObjectDetector(cfg=cfg)

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
    IMGS = os.listdir(IMGS_DIR)
    IMGS.sort()
    is_out = False
    for i in IMGS:
        if os.path.splitext(i)[-1].lower() not in ['.jpg', '.png', '.jpeg', '.bmp']:
            continue
        img_file = os.path.join(IMGS_DIR, i)
        if is_out is True:
            break
        f = cv2.imread(img_file)

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
            x1, y1, x2, y2 = map(int, d[:4])
            w, h = x2-x1, y2-y1
            fc = f.copy()
            cv2.rectangle(fc, (x1, y1), (x2, y2), (96, 96, 216), thickness=2, lineType=cv2.LINE_AA)

            cv2.imshow('_', fc)
            category_id = 0
            k = cv2.waitKey(0)
            if k == ord('q'):
                print("-- CV2 Stop --")
                is_out = True
                break
            elif k == ord('f'):
                print("-- Click f --")
            elif k == ord('m'):
                print("-- Click m --")
                category_id = 1

            anno_info = {"id": anno_ids,
                         "image_id": img_ids,
                         "category_id": category_id,
                         "bbox": [x1, y1, w, h],
                         "area": w*h,
                         "segmentation": [],
                         "iscrowd": 0}
            basic_fmt["annotations"].append(anno_info)
            anno_ids += 1
        img_ids += 1

    with open(os.path.join(IMGS_DIR, "../annotations.json"), 'w') as outfile:
        json.dump(basic_fmt, outfile, indent=2)


if __name__ == "__main__":
    main()
