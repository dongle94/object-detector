# -*- coding: utf-8 -*-
"""
converting tool for datavoucher coco json to csv
"""

import json
import csv
import argparse


def main(opt):
    input_json = opt.input_json
    output_csv = opt.output_csv

    with open(input_json, 'r') as file:
        json_data = json.load(file)

    annos = json_data["annotations"]

    with open(output_csv, mode='w', newline='') as f:
        wr = csv.writer(f)
        # Header
        wr.writerow(['id', 'image_id', 'category_id', 'x', 'y', 'w', 'h', 'area'])

        # Contents
        for anno in annos:
            wr.writerow([
                anno['id'], anno['image_id'], anno['category_id'],
                anno['bbox'][0], anno['bbox'][1], anno['bbox'][2], anno['bbox'][3],
                anno['area']
            ])


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_json', required=True,
                        help='input json file')
    parser.add_argument('-o', '--output_csv', required=True,
                        help="output csv file")
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    args = args_parse()

    main(args)
