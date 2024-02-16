import os
import cv2

from core.medialoader import LoadSample


class LoadStream(LoadSample):
    def __init__(self, source):
        super().__init__()

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        pass

    def __len__(self):
        pass


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # p1 = './data/images/'
    # p2 = './data/images/*'
    # p3 = './data/images/sample.jpg'
    # loader = LoadImages(p1)
    # for _im in loader:
    #     plt.imshow(_im)
    #     plt.show()
    pass