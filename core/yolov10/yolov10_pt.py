import sys
import torch

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.yolov10.yolov10_utils.torch_utils import select_device
from core.yolov10.nn.tasks import attempt_load_weights


class Yolov10Torch(object):
    def __init__(self, weight: str, device: str = 'cpu', fp16: bool = False, fuse: bool = True,
                 img_size: int = 640,):
        super(Yolov10Torch, self).__init__()
        self.device = select_device(device=device)
        self.cuda = torch.cuda.is_available() and device != "cpu"
        if fp16 is True and self.device.type != "cpu":
            self.fp16 = True
        else:
            self.fp16 = False

        model = attempt_load_weights(weight, device=self.device, inplace=True, fuse=fuse)
        model.half() if self.fp16 else model.float()
        for p in model.parameters():
            p.requires_grad = False
        self.model = model
        self.model.eval()
        self.stride = max(int(model.stride.max()), 32)
        self.names = model.module.names if hasattr(model, "module") else model.names
        # self.imgsz = check_imgsz(img_size, stride=self.model.stride, min_dim=2)

    def warmup(self):
        pass

    def preprocess(self, img):
        pass


if __name__ == "__main__":
    from utils.config import set_config, get_config

    set_config('./configs/config.yaml')
    cfg = get_config()

    yolov10 = Yolov10Torch(cfg.det_model_path, device=cfg.device, fp16=cfg.det_half,
                           img_size=cfg.yolov8_img_size)
    yolov10.warmup()
