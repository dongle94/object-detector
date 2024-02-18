import os
import cv2
import math
import platform

from core.medialoader import LoadSample

VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'webm'


class LoadVideo(LoadSample):
    def __init__(self, path, stride=1, opt=None):
        super().__init__()

        self.stride = stride

        if path.split('.')[-1].lower() not in VID_FORMATS:
            raise FileNotFoundError(f"File ext is invalid: {path}")

        if platform.system() == 'Windows':
            cap = cv2.VideoCapture(path, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(path)
        assert cap.isOpened(), f'Failed to open {path}'

        self.mode = 'video'

        # TODO: Check valid
        if opt is not None and opt.media_opt_auto is False:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, opt.media_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, opt.media_height)
            cap.set(cv2.CAP_PROP_FPS, opt.media_fps)
        self.w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback
        self.frame = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        # if self.count == self.num_files:
        #     raise StopIteration
        #
        # path = self.files[self.count]
        #
        # self.count += 1
        # im = cv2.imread(path)
        # assert im is not None, f'Image Not Found {path}'
        # im = im[..., ::-1]
        #
        # return im
        pass

    def __len__(self):
        pass


if __name__ == "__main__":
    p1 = './data/images/'
    loader = LoadVideo(p1)
    for _im in loader:
        _im = _im[..., ::-1]
        cv2.imshow('.', _im)
