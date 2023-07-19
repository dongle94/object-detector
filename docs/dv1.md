
# DV1

## Install 
### requirement
- Python 3.8 (test version)
- Windows or Linux OS
  - Test OS: Windows 10, Ubuntu 20.04

```shell
# install python package
$ pip install -r ./requirements/dv1.txt
```

## Weight Downloads
- yolov5 release 7.0 weight 
  - https://github.com/ultralytics/yolov5/releases/tag/v7.0
  - yolo weight contains each classes' name.
- osnet(tracking)
  - https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO


## run
`dv1.yaml` include yolo configs and tracker configs. 
Modify the configuration file
to fit your needs.

```shell
# repository root directory
$ python gui/dv1.py -c ./configs/dv1.yaml
```