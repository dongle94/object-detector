# -*- coding: utf-8 -*-
"""
Convert COCO json file to YOLO txt file
"""


import json
import os
import sys
import argparse
import yaml
import shutil


def main(opt):
    json_file = opt.json
    root_dir = opt.root_dir
    data_type = opt.type
    output_dir = opt.output_dir

    if not os.path.exists(json_file):
        print("Json file not exist.")
        return

    with open(json_file, 'r') as file:
        label_data = json.load(file)

    images_dir = os.path.join(output_dir, data_type, 'images')
    labels_dir = os.path.join(output_dir, data_type, 'labels')
    print(f"imaged_dir: {images_dir}")
    print(f"labels_dir: {labels_dir}")

    # create yolo dir(once)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    img_num = 0
    anno_num = 0
    for anno in label_data['annotations']:
        img_id = anno["image_id"]
        bbox = anno['bbox']
        img_info = [image for image in label_data["images"] if image["id"] == img_id][0]
        img_path = img_info["file_name"]

        orig_img_path = os.path.join(root_dir, img_path)
        full_img_path = os.path.join(output_dir, data_type, 'images', os.path.basename(img_path))

        # copy img file(once)
        if not os.path.exists(full_img_path):
            shutil.copy2(orig_img_path, full_img_path)
            img_num += 1

        # append labeling yolo txt file
        label_file = os.path.basename(img_path)
        label_file_path = os.path.join(labels_dir, os.path.splitext(label_file)[0] + '.txt')
        # {class x_center y_center width height}
        cls_id = anno["category_id"]
        img_w, img_h = img_info["width"], img_info["height"]
        x, y, w, h = anno['bbox']

        center_x, center_y = (x + w / 2) / img_w, (y + h / 2) / img_h
        rel_w, rel_h = w / img_w, h / img_h

        with open(label_file_path, 'a') as outfile:
            outfile.write(f"{cls_id} {center_x} {center_y} {rel_w} {rel_h}\n")
        anno_num += 1

    print(f"Converting success {img_num} images, {anno_num} annotations.")


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--root_dir', required=True,
                        help="root image directory")
    parser.add_argument('-j', '--json', required=True,
                        help='labeling json file path')
    parser.add_argument('-t', '--type', default='train',
                        help='type is in [train, val]. this option write file_path {type}/img_file')
    parser.add_argument('-o', '--output_dir', required=True,
                        help='this directory will have train/val yolo dirs.')
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    args = args_parse()

    main(args)
