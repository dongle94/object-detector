# Object-Detector
Object Detection based Project


## Installation
- python 3.8.x
- recommend conda envs

```shell
$ pip install -r ./requirements.txt 
```

## Modules
- medialoader
- detectors
  - yolov5
  - yolov8
- object_tracker
  - osnet

## Tools
### Human head mosaic
Using yolo head & people detector model and dlib face detection.
- I referred to project [yolov5-crowdhuman](https://github.com/deepakcrk/yolov5-crowdhuman) 
  - yolo crowd detection model download [Link](https://drive.google.com/file/d/1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb/view)
  - And i also make release with model weight file.

- Dlib Face detection model
  - model weight file in `weight` directory in this repo.
