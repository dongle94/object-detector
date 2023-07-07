import copy
import os
import sys
import numpy as np

import onnxruntime as ort

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
os.chdir(ROOT)

from obj_detectors.models.image_processing import letterbox, non_max_suppression, scale_boxes

class YoloOnnxDetector(object):
    def __init__(self, weight, device='cpu', img_size=640):
        self.imgsz = img_size

        # Execution Provider
        exec_provider = [
            'CPUExecutionProvider'
        ]
        if type(device) == int:
            _cuda_provider = ('CUDAExecutionProvider', {
                'device_id': device,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'HEURISTIC',
                'do_copy_in_default_stream': True,
            })
            exec_provider.insert(0, _cuda_provider)

        self.sess = ort.InferenceSession(weight, providers=exec_provider)

        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [output.name for output in self.sess.get_outputs()]
        self.output_shapes = [output.shape for output in self.sess.get_outputs()]
        self.output_types = [output.type for output in self.sess.get_outputs()]

        self.meta = self.sess.get_modelmeta().custom_metadata_map
        if 'stride' in self.meta:
            self.stride, self.names = int(self.meta['stride']), eval(self.meta['names'])

    def warmup(self, imgsz=(1, 3, 640, 640)):
        _im = np.zeros(imgsz, dtype=np.float32)
        self.infer(_im)

    def preprocess(self, img):
        _img = letterbox(im=img, new_shape=self.imgsz)[0]

        _img = _img.transpose((2, 0, 1))[::-1]
        _img = np.ascontiguousarray(_img)
        _img = _img.astype(np.float32)
        _img /= 255.0
        if len(_img.shape) == 3:
            _img = np.expand_dims(_img, axis=0)

        return _img, img

    def infer(self, img):
        ret = self.sess.run(self.output_names, {self.input_name: img})

        return ret

    def postprocess(self, pred, im_shape, im0_shape):
        pred = non_max_suppression(preds=pred)[0]
        det = scale_boxes(im_shape[2:], copy.deepcopy(pred[:, :4]), im0_shape).round()
        det = np.concatenate([det, pred[:, 4:]], axis=1)

        return pred, det



if __name__ == "__main__":
    model = YoloOnnxDetector(weight="./weights/yolov5s6.onnx",
                             device=0,
                             img_size=1280)
    model.warmup(imgsz=(1, 3, 1280, 1280))

    import cv2
    im0 = cv2.imread('./data/images/army.jpg')

    im, im0 = model.preprocess(im0)
    _pred = model.infer(im)
    _pred, _det = model.postprocess(_pred, im.shape, im0.shape)

    for d in _det:
        print(f"box:{int(d[0]), int(d[1]), int(d[2]), int(d[3])}, class: {model.names[int(d[5])]}, conf: {d[4]:.3f}")

        x1, y1, x2, y2 = map(int, d[:4])
        cv2.rectangle(im0, (x1, y1), (x2, y2), (128, 128, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(im0, f"{model.names[int(d[-1])]}: {d[-2]:.2f}", (x1, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 255), 1)


    cv2.imshow("_", im0)
    cv2.waitKey(0)
