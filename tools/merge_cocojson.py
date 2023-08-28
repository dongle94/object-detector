# -*- coding: utf-8 -*-
"""
COCO label json file merge tool
"""

import json
import os
import sys
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
                'cattegories': meta_json['categories'],
                "images": [],
                "annotations": []
            }
        img_ids = 0
        anno_ids = 0
    else:
        print("Meta json file not exist")
        return

    for input_json in inputs_json:
        imageid_map = {}
        if os.path.exists(input_json):
            with open(input_json, 'r') as file:
                json_dict = json.load(file)
            for json_img in json_dict['images']:
                imageid_map[img_ids] = json_img['id']
                json_img['id'] = img_ids
                basic_fmt['images'].append(json_img)
                img_ids += 1

            for json_anno in json_dict['annotations']:
                image_id = getKeybyValue(imageid_map, json_anno['image_id'])
                json_anno['id'] = anno_ids
                basic_fmt['annotations'].append(json_anno)
                anno_ids += 1

    print(basic_fmt)
    with open(opt.output, 'w') as outfile:
        json.dump(basic_fmt, outfile, indent=2, ensure_ascii=False)
    print("Merge success. stop program.")

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputs', required=True, nargs='+',
                        help='input json file path, use 1st input json metadata(info, license, categories)')
    parser.add_argument('-o', '--output', required=True,
                        help='output json file path')
    _args = parser.parse_args()
    return _args

if __name__ == "__main__":
    args = args_parse()

    main(args)