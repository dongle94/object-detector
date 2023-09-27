# -*- coding: utf-8 -*-
"""
COCO label json file merge tool
"""

import os
import json
import copy
import argparse


def getKeybyValue(cls_dict, idx):
    for k, v in cls_dict.items():
        if v == idx:
            return k


def main(opt):
    inputs_json = opt.inputs
    if len(inputs_json) == 0:
        print("No input json files")
        return

    if os.path.exists(inputs_json[0]):
        with open(inputs_json[0], 'r') as file:
            meta_json = json.load(file)
            basic_fmt = {
                'info': meta_json['info'],
                'licenses': meta_json['licenses'],
                'categories': meta_json['categories'],
                "images": [],
                "annotations": []
            }
        img_ids = 0
        anno_ids = 0
    else:
        print("Meta json file not exist")
        return

    for input_json in inputs_json:
        if not os.path.exists(input_json):
            print(f"No exist {input_json}")
            continue

        print(f"Start {input_json}")
        imageid_map = {}
        with open(input_json, mode='r', encoding='utf8') as file:
            json_dict = json.load(file)
        # Process(Merge) images field
        for json_img in json_dict['images']:
            imageid_map[img_ids] = json_img['id']
            img_json = copy.deepcopy(json_img)
            img_json['id'] = img_ids
            basic_fmt['images'].append(img_json)
            img_ids += 1

        # Process(Merge) annotations field
        for json_anno in json_dict['annotations']:
            orig_img_id = json_anno['image_id']
            new_img_id = getKeybyValue(imageid_map, orig_img_id)

            anno_json = copy.deepcopy(json_anno)
            anno_json['id'] = anno_ids
            anno_json['image_id'] = new_img_id
            basic_fmt['annotations'].append(anno_json)
            anno_ids += 1

        print(f"End {input_json}")

    with open(opt.output, 'w') as outfile:
        json.dump(basic_fmt, outfile, indent=1, ensure_ascii=False)
    print(f"Merge success to {opt.output}. stop program.")


def args_parse():
    parser = argparse.ArgumentParser(description="First input json's metadata is leading whole metadata.")
    parser.add_argument('-i', '--inputs', required=True, nargs='+',
                        help='input json file path, use 1st input json metadata(info, license, categories)')
    parser.add_argument('-o', '--output', required=True,
                        help='output json file path')
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    args = args_parse()

    main(args)
