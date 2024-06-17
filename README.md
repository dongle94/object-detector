# Object-Detector
Object Detection based Project
- Human and coco dataset detection
- Hand detection for hand pose estimation


## Installation
- Ubuntu 20.04, 22.04 (test version)
- python 3.8.18 (test version)
- CUDA 11.8 (test version)
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
  - yolov5
    - pytorch, onnx, trt
  - yolov8
    - pytorch, onnx, trt
- ~~object_tracker~~
  - ~~osnet~~

## Tools
### Convert module
- Pytorch to Onnx
- Onnx to TensorRT

## Reference
- YOLOv5: https://github.com/ultralytics/yolov5
- YOLOv8(ultraytics): https://github.com/ultralytics/ultralytics

[//]: # (### Human head mosaic)

[//]: # (Using yolo head & people detector model and dlib face detection.)

[//]: # (- I referred to project [yolov5-crowdhuman]&#40;https://github.com/deepakcrk/yolov5-crowdhuman&#41; )

[//]: # (  - yolo crowd detection model download [Link]&#40;https://drive.google.com/file/d/1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb/view&#41;)

[//]: # (  - And i also make release with model weight file.)

[//]: # ()
[//]: # (- Dlib Face detection model)

[//]: # (  - model weight file in `weight` directory in this repo.)
