import os
import cv2
import math
import platform

from core.medialoader import LoadSample

VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'webm'


class LoadVideo(LoadSample):
    def __init__(self, path, stride=1):
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

        self.w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.cap = cap
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback
        self.frame = 0
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.stride)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            raise StopIteration

        for _ in range(self.stride):
            self.cap.grab()
        ret, im = self.cap.retrieve()
        while not ret:
            self.cap.release()
            raise StopIteration

        self.frame += 1
        im = im[..., ::-1]

        return im

    def __len__(self):
        pass


if __name__ == "__main__":
    p1 = './data/videos/sample.mp4'
    loader = LoadVideo(p1)
    for _im in loader:
        _im = _im[..., ::-1]
        cv2.imshow('.', _im)
