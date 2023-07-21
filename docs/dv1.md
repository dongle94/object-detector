
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

## deployment
To create linux executable and Windows exe program need `Pyinstaller` library.
you need install pyinstaller.
```shell
$ pip install pyinstaller
```
And below scripts create executable program with no detail install.
```shell
$ pyinstaller -D ./gui/dv1.py --name DV1 -p ./ -p ./obj_detectors/ --collect-all torchvision
```
It makes `build`, `dist` directory in repository. we use `dist` directory. 
That directory includes executable file you naming with `--name`.
Before you run, you need to copy weight and config in this directory.
you can execute this file in terminal or by double click.
```shell
$ cd ./dist/dv1
$ cp -r ../../weights ./
$ cp -r ../../configs ./

# default is ./DV1 -c ./configs/dv1.yaml
$ ./DV1 
```