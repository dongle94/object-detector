import sys

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.yolov10.nn.tasks import attempt_load_one_weight


class Yolov10Torch(object):
    def __init__(self, weight: str):
        super(Yolov10Torch, self).__init__()

        self.device = select_device(device=device)

        weight = str(weight).strip()
        if Path(weight).suffix in (".yaml", ".yml"):
            self._new(weight)
        else:       # .pt, .pth
            self._load(weight)

    def _new(self, weight: str):
        pass

    def _load(self, weight: str):
        self.model, self.ckpt = attempt_load_one_weight(weight)


if __name__ == "__main__":
    from utils.config import set_config, get_config

    set_config('./configs/config.yaml')
    cfg = get_config()

    yolov10 = Yolov10Torch()
