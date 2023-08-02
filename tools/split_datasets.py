import os
import sys
import argparse
import shutil
from tqdm import tqdm

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
os.chdir(ROOT)

def run(opt):
    image_dirs = opt.image_dir
    print(image_dirs)
    for image_dir in image_dirs:
        # extract list for only images.
        # os.path.splitext(i)[1] == pathlib.Path(i).suffix
        imgs = [i for i in os.listdir(image_dir) if Path(i).suffix.lower() in ['.jpg', '.png', '.jpeg', '.bmp']]
        imgs.sort()

        # check train/val ratio and number
        imgs_num = len(imgs)
        train_num = int(imgs_num * opt.tv_ratio)
        val_num = imgs_num - train_num
        print(f"{image_dir} dir has {imgs_num} images. it will split images into train: {train_num} / val: {val_num}")

        # create directory for train/val
        train_path = os.path.join(image_dir, 'train')
        val_path = os.path.join(image_dir, 'val')
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        if not os.path.exists(val_path):
            os.makedirs(val_path)

        # split and copy images for train/val
        train_imgs = imgs[:][:train_num]
        val_imgs = imgs[:][train_num:]
        t_timgs = tqdm(train_imgs, 'split train_images')
        for train_img in t_timgs:
            src_path = os.path.join(image_dir, train_img)
            dst_path = os.path.join(image_dir, 'train', train_img)
            if opt.move:
                shutil.move(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)
        t_vimgs = tqdm(val_imgs, 'split val_images')
        for val_img in t_vimgs:
            src_path = os.path.join(image_dir, val_img)
            dst_path = os.path.join(image_dir, 'val', val_img)
            if opt.move:
                shutil.move(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

    print("Success split datasets!")


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_dir', required=True, nargs='+',
                        help='directories for images to split training sets and tests sets')
    parser.add_argument('-r', '--tv_ratio', type=float, default=0.9,
                        help="training set ratio. default is 0.9. "
                             "0.9 means that training sets' ratio is 0.9 and validation sets' 0.1")
    parser.add_argument('--move', action='store_true',
                        help='if write this option, no image copy, move images.')
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    args = args_parse()
    run(args)
