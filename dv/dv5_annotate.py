# -*- coding: utf-8 -*-
"""
datavoucher 5 - quamtomai
human group object classes by gender: homo, hetero, unknown
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

        cls_1 = boxes[p[0]][5]
        cls_2 = boxes[p[1]][5]
        group_cls = 1 if cls_1 == cls_2 else 2

        new_boxes.append((x1, y1, x2, y2, group_cls))
    return new_boxes


def draw_event(event, x, y, flags, param):
    global mouseX, mouseY, img, box_point, label_info
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        cv2.circle(img, (x, y), 2, (32, 216, 32), -1)
        box_point.append((x, y))
        if len(box_point) == 2:
            box_pt1, box_pt2 = get_box_point(box_point[0], box_point[1])
            crop_img = img[box_pt1[1]:box_pt2[1], box_pt1[0]:box_pt2[0]]
            cv2.imshow("crop", crop_img)
            key = cv2.waitKey(0)

            if key == ord("1"):          # homo
                _class = 1
                b_color = (248, 16, 16)
            elif key == ord("2"):       # hetero
                _class = 2
                b_color = (16, 16, 248)
            elif key == ord("3"):       # unknown
                _class = 3
                b_color = (16, 248, 16)
            else:
                box_point = []
                cv2.destroyWindow("crop")
                return
            label_info.append((box_pt1, box_pt2, _class))
            cv2.rectangle(img, box_pt1, box_pt2, b_color, 1, cv2.LINE_AA)
            box_point = []
            cv2.destroyWindow("crop")
            cv2.imshow(param, img)

    if event == cv2.EVENT_MOUSEMOVE:
        im = img.copy()
        img_size = im.shape
        cv2.line(im, (x, 0), (x, img_size[0]), (0, 0, 0), 1, cv2.LINE_AA)
        cv2.line(im, (0, y), (img_size[1], y), (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow(param, im)

def main(opt=None):
    get_logger().info(f"Start dv5 annotation script.")
    IMGS_DIRS = opt.imgs_dir
    get_logger().info(f"Input Directory is {IMGS_DIRS}")

    detector = ObjectDetector(cfg=cfg)

    obj_classes = defaultdict(int)

    img_ids = 0
    anno_ids = 0
    if os.path.exists(opt.json_file):
        with open(opt.json_file, 'r') as file:
            basic_fmt = json.load(file)
        get_logger().info(f"{opt.json_file} is exist. append annotation file.")
        if len(basic_fmt['images']) != 0:
            img_ids = int(basic_fmt['images'][-1]['id']) + 1
            get_logger().info(f"last image file name: {basic_fmt['images'][-1]['file_name']}")
        if len(basic_fmt['annotations']) != 0:
            anno_ids = int(basic_fmt["annotations"][-1]['id']) + 1
            for anno in basic_fmt['annotations']:
                obj_classes[int(anno['category_id'])] += 1
            get_logger().info(f"old object classes: {obj_classes}")
    else:
        get_logger().info(f"{opt.json_file} is not exist. Create new annotation file")
        basic_fmt = {
            "info": {"year": "2023", "version": "1",
                     "description": "datavoucher group people object detection ",
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

    is_out = False
    image_extension = ['.jpg', '.png', '.jpeg', '.bmp']

    for IMGS_DIR in IMGS_DIRS:
        if is_out is True:
            break

        IMGS = [i for i in os.listdir(IMGS_DIR) if os.path.splitext(i)[-1].lower() in image_extension]
        IMGS.sort()
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
            _arr = get_l2_norm(_det)
            _boxes = get_closed_boxes(_arr, threshold=120)
            _new_boxes = get_merge_boxes(_det, _boxes)
            get_logger().info(f"pair boxes: {_boxes}")

            c_id = defaultdict(int)
            tmp_annos = []

            f1 = f0.copy()
            # draw whole boxes in images
            for idx, d in enumerate(_det):
                x1, y1, x2, y2 = map(int, d[:4])
                _cls_idx = int(d[5])
                b_color = (128, 255, 128)
                if detector.names[_cls_idx] == 'female':
                    b_color = (192, 192, 255)
                elif detector.names[_cls_idx] == 'male':     # male
                    b_color = (255, 192, 192)

                cv2.rectangle(f1, (x1, y1), (x2, y2), b_color, thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(f1, f"{idx}", (x1+10, y1+20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=(255, 16, 16), thickness=2)

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

            # draw group box
            auto_box = 0
            for _idx, d in enumerate(_new_boxes):
                f2 = f1.copy()
                x1, y1, x2, y2 = d[:4]
                cls = int(d[4])
                b_color = (16, 255, 16)
                if cls == 1:    # homo(동성그룹)   초록색
                    b_color = (48, 232, 48)
                elif cls == 2:  # hetero(이성그룹) 노랑색
                    b_color = (232, 232, 48)
                w, h = x2-x1, y2-y1

                nx1, ny1 = cvtPoint(x1, y1, orig_img_size, edit_img_size)
                nx2, ny2 = cvtPoint(x2, y2, orig_img_size, edit_img_size)
                cv2.rectangle(f2, (nx1, ny1), (nx2, ny2), b_color, thickness=2, lineType=cv2.LINE_AA)
                img = f2
                cv2.imshow(winname, f2)

                new_frame = f0[y1:y2, x1:x2]
                b_name = f"{_idx + 1}/{len(_new_boxes)}"
                cv2.imshow(b_name, new_frame)

                category_id = 0
                _k = cv2.waitKey(0)

                if _k == ord('q'):
                    get_logger().info("-- CV2 Stop --")
                    is_out = True
                    cv2.destroyWindow(b_name)
                    cv2.imshow(winname, f1)
                    img = f1
                    break
                elif _k == ord('1'):     # homo
                    category_id = 1
                    b_color = (248, 16, 16)
                elif _k == ord('2'):     # hetero
                    category_id = 2
                    b_color = (16, 16, 248)
                elif _k == ord('3'):     # unknown
                    category_id = 3
                    b_color = (16, 248, 16)
                else:
                    cv2.destroyWindow(b_name)
                    img = f1
                    cv2.imshow(winname, f1)
                    continue
                auto_box += 1
                cv2.rectangle(f1, (nx1, ny1), (nx2, ny2), b_color, thickness=1, lineType=cv2.LINE_AA)
                img = f1
                cv2.imshow(winname, f1)
                cv2.destroyWindow(b_name)

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

            k = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if k == ord('q'):
                get_logger().info("-- CV2 Stop --")
                is_out = True
                break
            elif k == ord(" "):
                for l_info in label_info:
                    pt1, pt2 = l_info[0], l_info[1]
                    x1, y1 = cvtPoint(pt1[0], pt1[1], edit_img_size, orig_img_size)
                    x2, y2 = cvtPoint(pt2[0], pt2[1], edit_img_size, orig_img_size)
                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    anno_info = {
                        "id": anno_ids,
                        "image_id": img_ids,
                        "category_id": l_info[2],
                        "bbox": [x1, y1, w, h],
                        "area": w * h,
                        "segmentation": [],
                        "iscrowd": 0
                    }
                    anno_ids += 1
                    tmp_annos.append(anno_info)
                    c_id[l_info[2]] += 1

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

    with open(os.path.join(opt.json_file), 'w') as outfile:
        json.dump(basic_fmt, outfile, indent=1)

    get_logger().info(f"obj classes: {obj_classes}")


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--imgs_dir', required=True, nargs='+',
                        help='image directory path')
    parser.add_argument('-j', '--json_file', required=True,
                        help="if write this file, append annotations")
    parser.add_argument('-t', '--type', default='train',
                        help='type is in [train, val]. this option write file_path {type}/img_file')
    parser.add_argument('-c', '--config', default='./configs/annotate.yaml',
                        help="annotation configuration yaml file path")
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    args = args_parse()

    update_config(cfg, args=args.config)
    init_logger(cfg)

    main(args)
