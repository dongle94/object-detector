# Object-Detector
Object Detection based Project
- Human and coco dataset detection
- Hand detection for hand pose estimation


## Installation
- Ubuntu 20.04, 22.04 (test version)
- python >= 3.8.x
  - test: 3.8.18 / 3.10.14
- CUDA >= 11.8 (test version)
- recommend conda envs
- torch 2.0.1 (test version)
- torchvision 0.15.2 (test version)
- onnx >= 1.15.x
- onnxruntime-gpu
- tensorrt-cu11 10.0.1 (test version)

```shell
$ pip install -r ./requirements.txt 
```

## Modules
- medialoader
- detectors
  - YOLOv5
    - pytorch, onnx, trt
  - YOLOv8
    - pytorch, onnx, trt
  - YOLOv10
    - pytorch
- ~~object_tracker~~
  - ~~osnet~~

## Reference
- YOLOv5: https://github.com/ultralytics/yolov5
- YOLOv8(ultraytics): https://github.com/ultralytics/ultralytics
- YOLOv10: https://github.com/THU-MIG/yolov10