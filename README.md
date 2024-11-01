# Object-Detector
Object Detection based Project
- Human and coco dataset detection
- Hand detection for hand pose estimation


## Installation
- Ubuntu 20.04, 22.04 (test version)
- python >= 3.8.x
  - test: 3.8.x / 3.10.x
- CUDA >= 11.8 (test version)
  - test: 11.8 / 12.x
  - recommend using cuda-python package
- recommend conda envs
- Pytorch
  - torch 2.x (test version)
    - test: 2.0.1 / 2.4.1
  - torchvision: torch version compatibility
- Optimization
  - onnx >= 1.15.x
  - onnxruntime-gpu
  - tensorrt-cu12 10.0.1 (test version)

```shell
$ pip install -r ./requirements.txt 
```

## Modules
- medialoader
- detectors
  - YOLOv5, YOLOv8, YOLOv10
    - pytorch, onnx, trt
- ~~object_tracker~~
  - ~~osnet~~

## Reference
- YOLOv5: https://docs.ultralytics.com/models/yolov5/
- YOLOv8: https://docs.ultralytics.com/models/yolov8/
- YOLOv10: https://github.com/THU-MIG/yolov10
  - https://docs.ultralytics.com/models/yolov10/