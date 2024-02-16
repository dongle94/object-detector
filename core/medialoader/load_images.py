import os
import cv2

from core.medialoader import LoadSample

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'


class LoadImages(LoadSample):
    def __init__(self, path):
        super().__init__()
        files = []
        path = sorted(path) if isinstance(path, (list, tuple)) else [path]
        for p in path:
            p = os.path.abspath(p)
            if '*' in p:
                files = [os.path.join(os.path.dirname(p), f) for f in os.listdir(os.path.dirname(p))]
            elif os.path.isdir(p):
                files = [os.path.join(p, f) for f in os.listdir(p)]
            elif os.path.isfile(p):
                files.append(p)
            else:
                FileNotFoundError(f'{p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        ni = len(images)

        self.files = images
        self.num_files = ni

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.num_files:
            raise StopIteration

        path = self.files[self.count]

        self.count += 1
        im = cv2.imread(path)
        assert im is not None, f'Image Not Found {path}'
        im = im[..., ::-1]

        return im

    def __len__(self):
        return self.num_files  # number of files


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    p1 = './data/images/'
    p2 = './data/images/*'
    p3 = './data/images/sample.jpg'
    loader = LoadImages(p1)
    for _im in loader:
        plt.imshow(_im)
        plt.show()
