import os
import sys
import argparse

import torch
import onnx

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
os.chdir(ROOT)

from obj_detectors.yolov5_pt import attempt_load
from obj_detectors.models.torch_utils import select_device
from obj_detectors.models.yolo import Detect

def convert(opt):
    # input shape
    imgsz = opt.imgsz * 2 if len(opt.imgsz) == 1 else opt.imgsz
    device = select_device(opt.device)
    file = Path(opt.weight)
    f = file.with_suffix('.onnx')
    fp16 = opt.fp16

    im = torch.zeros(1, 3, *imgsz).to(device)

    _model = attempt_load(opt.weight, device=device, inplace=True, fuse=True)   # load FP32 model
    _model.eval()
    for k, m in _model.named_modules():
        if isinstance(m, Detect):
            m.inplace = False
            m.dynamic = False
            m.export = True

    for _ in range(2):
        y = _model(im)  # dry runs
    if fp16:
        im, _model = im.half(), _model.half()
    shape = tuple((y[0] if isinstance(y, tuple) else y).shape)

    print(f"Converting from {file} with output shape {shape}")

    #f[2], _ = export_onnx(model, im, file, opset, dynamic, simplify)

    output_names = ['output0']

    torch.onnx.export(
        _model,
        im,
        f,
        verbose=False,
        opset_version=opt.opset,
        do_constant_folding=False,
        input_names=['images'],
        output_names=output_names,
        dynamic_axes=None
    )

    # check
    model_onnx = onnx.load(f)   # load onnx model
    onnx.checker.check_model(model_onnx)    # check onnx model

    # metadata
    d = {'stride': int(max(_model.stride)), 'names': _model.names}
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)

    # Simplify
    if opt.simplify:
        try:
            cuda = torch.cuda.is_available()
            import onnxsim

            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, "assert check failed"
        except Exception as e:
            print(f"simplifier failure: {e}")

    onnx.save(model_onnx, f)

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight', required=True, help=".pt or .pth file path")
    parser.add_argument('-d', '--device', default='cpu', help='cpu or 0, 1, 2,...')
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640, 640], help="image (h, w)",
                        choices=[640, 1280, [640, 640], [1280, 1280]])
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=18, help='ONNX: opset version')
    parser.add_argument('--fp16', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parse()
    convert(args)
