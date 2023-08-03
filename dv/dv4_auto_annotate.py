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

import glob
import tensorflow as tf
from tqdm import tqdm


def crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width):
    cropped = image[offset_height:offset_height + target_height, offset_width:offset_width + target_width, :]
    return cropped


def pad_to_bounding_box(image, offset_height, offset_width, target_height, target_width):
    height, width, depth = image.shape

    after_padding_width = target_width - offset_width - width

    after_padding_height = target_height - offset_height - height
    # Do not pad on the depth dimensions.
    paddings = ((offset_height, after_padding_height), (offset_width, after_padding_width), (0, 0))
    padded = np.pad(image, paddings, 'constant')

    return padded


def resize_image_with_crop_or_pad(image, target_height, target_width):
    # crop to ratio, center
    height, width, c = image.shape

    width_diff = target_width - width
    offset_crop_width = max(-width_diff // 2, 0)
    offset_pad_width = max(width_diff // 2, 0)

    height_diff = target_height - height
    offset_crop_height = max(-height_diff // 2, 0)
    offset_pad_height = max(height_diff // 2, 0)

    # Maybe crop if needed.
    # print('image shape', image.shape)
    cropped = crop_to_bounding_box(image, offset_crop_height, offset_crop_width,
                                   min(target_height, height),
                                   min(target_width, width))
    # print('after cropp', cropped.shape)
    # Maybe pad if needed.
    resized = pad_to_bounding_box(cropped, offset_pad_height, offset_pad_width,
                                  target_height, target_width)
    # print('after pad', resized.shape)
    return resized[:target_height, :target_width, :]


def read_prep_image(im_fname, avoid_distortion=True):
    '''
    if min(height, width) is larger than 224 subsample to 224. this will also affect the larger dimension.
    in the end crop and pad the whole image to get to 224x224
    :param avoid_distortion:
    :param im_fname:
    :return:
    '''

    if isinstance(im_fname, np.ndarray):
        image_data = im_fname
    else:
        image_data = cv2.imread(im_fname, 3)

    height, width = image_data.shape[0], image_data.shape[1]

    if avoid_distortion:
        if max(height, width) > 224:
            # print(image_data.shape)
            rt = 224. / max(height, width)
            image_data = cv2.resize(image_data, (int(rt * width), int(rt * height)), interpolation=cv2.INTER_AREA)
            # print('>>resized to>>',image_data.shape)
    else:
        image_data = cv2.resize(image_data, (224, 224), interpolation=cv2.INTER_LINEAR)

    image_data = resize_image_with_crop_or_pad(image_data, 224, 224)

    return image_data.astype(np.uint8)


def put_text_in_image(images, text, color=(0, 0, 0), position=None):
    '''
    :param images: 4D array of images
    :param text: list of text to be printed in each image
    :param color: the color or colors of each text
    :param position:
    :return:
    '''
    fontColors = {'red': (0, 0, 255),
                  'green': (0, 255, 0),
                  'yellow': (0, 255, 255),
                  'blue': (255, 0, 0),
                  'orange': (0, 165, 255),
                  'black': (0, 0, 0),
                  'grey': (169, 169, 169),
                  'white': (255, 255, 255),
                  }
    if not isinstance(text, list): text = [text]
    # if not isinstance(color, list): color = [color for _ in range(images.shape[0])]
    if images.ndim == 3: images = images.reshape(1, images.shape[0], images.shape[1], 3)
    images_out = []
    for imIdx in range(images.shape[0]):
        img = images[imIdx].astype(np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        if position is None: position = (10, img.shape[1])
        fontScale = 0.5
        lineType = 2
        fontColor = color
        cv2.putText(img, text[imIdx],
                    position,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        images_out.append(img)
    return np.array(images_out)


class Homogenus(object):
    def __init__(self, trained_model_dir, sess=None):
        best_model_fname = sorted(glob.glob(os.path.join(trained_model_dir, '*.ckpt.index')), key=os.path.getmtime)
        if len(best_model_fname):
            self.best_model_fname = best_model_fname[-1].replace('.index', '')
        else:
            raise ValueError('Couldnt find TF trained model in the provided directory --trained_model_dir=%s. '
                             'Make sure you have downloaded them there.' % trained_model_dir)

        if sess is None:
            self.sess = tf.compat.v1.Session()
        else:
            self.sess = sess

        # Load graph.
        tf.compat.v1.disable_eager_execution()
        self.graph = tf.compat.v1.get_default_graph()
        self.saver = tf.compat.v1.train.import_meta_graph(self.best_model_fname + '.meta')
        print(f'Restoring checkpoint {self.best_model_fname} ..')
        self.saver.restore(self.sess, self.best_model_fname)

        self.input = self.graph.get_tensor_by_name(u'input_images:0')
        self.output = self.graph.get_tensor_by_name(u'probs_op:0')

    def infer(self, imgs):

        probs_ob = self.sess.run(self.output, feed_dict={self.input: imgs}) #[0]
        gender_id = np.argmax(probs_ob, axis=1)

        ret = []
        for idx, _id in enumerate(gender_id):
            gender_prob = probs_ob[idx][_id]
            gender_name = 'male' if _id == 0 else 'female'
            ret.append([gender_name, gender_prob])

        return ret


def main(opt=None):
    IMGS_DIRS = opt.imgs_dir
    get_logger().info(f"Input Directory is {IMGS_DIRS}")

    detector = ObjectDetector(cfg=cfg)
    gender_classifier = Homogenus('./weights/homogenus_v1_0')

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
            "categories": [{"id": 0, "name": "male", "supercategory": "person"},
                           {"id": 1, "name": "female", "supercategory": "person"}],
            "images": [],
            "annotations": []
        }
        img_ids = 0
        anno_ids = 0

    crop_margin = opt.crop_margin
    gender_calib_threshold = opt.calib_threshold

    is_out = False
    for IMGS_DIR in IMGS_DIRS:
        if is_out is True:
            break

        show_img = True
        IMGS = os.listdir(IMGS_DIR)
        IMGS.sort()
        get_logger().info(f"process {IMGS_DIR} ..")
        _imgs = tqdm(IMGS, f'{IMGS_DIR} images ..')
        for i in _imgs:
            if is_out is True:
                break

            if os.path.splitext(i)[-1].lower() not in ['.jpg', '.png', '.jpeg', '.bmp']:
                continue
            img_file = os.path.join(IMGS_DIR, i)
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
            c_id = defaultdict(int)
            tmp_annos = []
            f1 = f0.copy()
            gender_meta = []
            gender_input = []
            for d in _det:
                if d[4] < float(opt.confidence):
                    continue
                x1, y1, x2, y2 = map(int, d[:4])
                w, h = x2 - x1, y2 - y1

                im_h, im_w = f0.shape[0], f0.shape[1]

                margin_h = crop_margin * im_h
                margin_w = crop_margin * im_w
                offset_h = int(max((y1 - margin_h), 0))
                target_h = int(min((y2 + margin_h), im_h)) - offset_h
                offset_w = int(max((x1 - margin_w), 0))
                target_w = int(min((x2 + margin_w), im_w)) - offset_w

                c_img0 = f0[offset_h:offset_h+target_h, offset_w:offset_w+target_w]
                c_img1 = read_prep_image(c_img0) #[np.newaxis]

                _g_inp = {
                    'coord': [x1, y1, x2, y2],
                    'width': w,
                    'height': h,
                }
                gender_meta.append(_g_inp)
                gender_input.append(c_img1)

            gender_input = np.array(gender_input)
            rets = gender_classifier.infer(gender_input)

            for g_meta, ret in zip(gender_meta, rets):
                x1, y1, x2, y2 = g_meta['coord']
                w, h = g_meta['width'], g_meta['height']
                if ret[0] == 'male':
                    txt_color = (255, 16, 16)
                    color = (255, 96, 96)
                    category_id = 0
                    if ret[1] < gender_calib_threshold:
                        ret = ['female', ret[1]]
                        txt_color = (16, 16, 255)
                        color = (96, 96, 255)
                        category_id = 1
                else:  # ret[0] == 'female'
                    txt_color = (16, 16, 255)
                    color = (96, 96, 255)
                    category_id = 1
                text = f'{ret[0]}[{ret[1]:.3f}]'
                if show_img is True:
                    cv2.rectangle(f1, (x1, y1), (x2, y2), color, thickness=2, lineType=cv2.LINE_AA)
                    f1 = put_text_in_image(f1, [text], txt_color, (x1, y1))[0]

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
                if f1.shape[0] > 1000:
                    f1 = cv2.resize(f1, (int(f1.shape[1] * 0.8), int(f1.shape[0] * 0.8)))
                cv2.imshow(img_file, f1)

                k = cv2.waitKey(0)
                cv2.destroyAllWindows()
                if k == ord('q'):
                    get_logger().info("-- CV2 Stop --")
                    break
                elif k == ord('p'):
                    continue
                elif k == ord('y'):
                    show_img = False

            for k, v in c_id.items():
                obj_classes[k] += v
            for _anno_info in tmp_annos:
                basic_fmt["annotations"].append(_anno_info)

            basic_fmt["images"].append(img_info)
            new_path = os.path.join(IMGS_DIR, 'already', i)
            # if not os.path.exists(os.path.dirname(new_path)):
            #     os.makedirs(os.path.dirname(new_path))
            # shutil.move(img_file, new_path)
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
    parser.add_argument('-c', '--confidence', default=0.3, type=float,
                        help="obj_detector confidence")
    parser.add_argument('-cm', '--crop_margin', default=0.04, type=float,
                        help="gender classification input image crop margin")
    parser.add_argument('-ct', '--calib_threshold', default=0.7, type=float,
                        help="gender classification score calibration threshold."
                             "if male score is lower than ct, label change to female.")
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    args = args_parse()

    update_config(cfg, args='./configs/annotate.yaml')
    init_logger(cfg)

    main(args)
