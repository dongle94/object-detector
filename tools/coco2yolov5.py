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
    t_json_file = opt.train_json
    v_json_file = opt.val_json
    root_dir = opt.root_dir
    output_dir = opt.output_dir

    if not os.path.exists(t_json_file):
        print("Train json file not exist.")
        return
    if not os.path.exists(v_json_file):
        print("Val json file not exist.")
        return

    with open(t_json_file, 'r') as file:
        t_label_data = json.load(file)
    with open(v_json_file, 'r') as file:
        v_label_data = json.load(file)

    t_images_dir = os.path.join(output_dir, 'train', 'images')
    t_labels_dir = os.path.join(output_dir, 'train', 'labels')
    v_images_dir = os.path.join(output_dir, 'val', 'images')
    v_labels_dir = os.path.join(output_dir, 'val', 'labels')
    print(f"imaged_dir: {t_images_dir}, {v_images_dir}")
    print(f"labels_dir: {t_labels_dir}, {v_labels_dir}")

    # Create yolo dir(once)
    if not os.path.exists(t_images_dir):
        os.makedirs(t_images_dir)
    if not os.path.exists(v_images_dir):
        os.makedirs(v_images_dir)
    if not os.path.exists(t_labels_dir):
        os.makedirs(t_labels_dir)
    if not os.path.exists(v_labels_dir):
        os.makedirs(v_labels_dir)

    # Create data.yaml
    meta_data = {}
    meta_data['path'] = output_dir
    meta_data['train'] = './train/images'
    meta_data['val'] = './val/images'

    meta_data['nc'] = len(t_label_data['categories'])
    meta_data['names'] = [category['name'] for category in t_label_data['categories']]

    print(yaml.dump(meta_data))

    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        yaml.dump(meta_data, f)

    # Train dataset process
    t_img_num = 0
    t_anno_num = 0
    for t_anno in t_label_data['annotations']:
        img_id = t_anno["image_id"]
        img_info = [image for image in t_label_data["images"] if image["id"] == img_id][0]
        img_path = img_info["file_name"]

        orig_img_path = os.path.join(root_dir, img_path)
        full_img_path = os.path.join(output_dir, 'train', 'images', os.path.basename(img_path))

        # copy img file(once)
        if not os.path.exists(full_img_path):
            shutil.copy2(orig_img_path, full_img_path)
            t_img_num += 1

        # append labeling yolo txt file
        label_file = os.path.basename(img_path)
        label_file_path = os.path.join(t_labels_dir, os.path.splitext(label_file)[0] + '.txt')
        # {class x_center y_center width height}
        cls_id = t_anno["category_id"]
        img_w, img_h = img_info["width"], img_info["height"]
        x, y, w, h = t_anno['bbox']

        center_x, center_y = (x + w / 2) / img_w, (y + h / 2) / img_h
        rel_w, rel_h = w / img_w, h / img_h

        with open(label_file_path, 'a') as outfile:
            outfile.write(f"{cls_id} {center_x} {center_y} {rel_w} {rel_h}\n")
        t_anno_num += 1

    # Val dataset process
    v_img_num = 0
    v_anno_num = 0
    for v_anno in v_label_data['annotations']:
        img_id = v_anno["image_id"]
        img_info = [image for image in v_label_data["images"] if image["id"] == img_id][0]
        img_path = img_info["file_name"]

        orig_img_path = os.path.join(root_dir, img_path)
        full_img_path = os.path.join(output_dir, 'val', 'images', os.path.basename(img_path))

        # copy img file(once)
        if not os.path.exists(full_img_path):
            shutil.copy2(orig_img_path, full_img_path)
            v_img_num += 1

        # append labeling yolo txt file
        label_file = os.path.basename(img_path)
        label_file_path = os.path.join(v_labels_dir, os.path.splitext(label_file)[0] + '.txt')
        # {class x_center y_center width height}
        cls_id = v_anno["category_id"]
        img_w, img_h = img_info["width"], img_info["height"]
        x, y, w, h = v_anno['bbox']

        center_x, center_y = (x + w / 2) / img_w, (y + h / 2) / img_h
        rel_w, rel_h = w / img_w, h / img_h

        with open(label_file_path, 'a') as outfile:
            outfile.write(f"{cls_id} {center_x} {center_y} {rel_w} {rel_h}\n")
        v_anno_num += 1

    print(f"Converting success Train: {t_img_num} images, {t_anno_num} annotations / "
          f"Val: {v_img_num} images, {v_anno_num} annotations ")


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--root_dir', required=True,
                        help="root image directory")
    parser.add_argument('-tj', '--train_json', required=True,
                        help='labeling json file path')
    parser.add_argument('-vj', '--val_json', required=True,
                        help='labeling json file path')
    parser.add_argument('-o', '--output_dir', required=True,
                        help='this directory will have train/val yolo dirs.')
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    args = args_parse()

    main(args)
