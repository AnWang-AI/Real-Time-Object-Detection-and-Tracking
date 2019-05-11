# Object-detecting-and-tracking
This project combines object detection and object tracking. The detector will detect the objects of the image captured by camera and the tracker will track the one of objects choosed by user. The detector is SSD model and tracker is SiamFPN model. Both models are real-time algorithms and you can use these algorithms only by CPU. You can run 'SSD+SiamRPN.py' to achieve object detecting and tracking.

If you are interested in the princile of these algorithms, please read these papaers:

* SSD: https://arxiv.org/abs/1512.02325
* SiamRPN: http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf

This repository includes original Tensorflow code of SSD
* https://github.com/wcwowwwww/SSD-Tensorflow 

This repository includes original pytorch code of SiamRPN
* https://github.com/wcwowwwww/DaSiamRPN.

### Prerequisites

* python
* numpy
* opencv
* tensorflow
* pytorch

### Pretrained model for SSD

You can download model from https://drive.google.com/file/d/0B0qPCUZ-3YwWUXh4UHJrd1RDM3c/view?usp=sharing , then please put this model into subfolder 'SSD/checkpoints', so that the detector can find and load the pretrained_model.

### Pretrained model for SiamRPN

You can download model from https://drive.google.com/drive/folders/1BtIkp5pB6aqePQGlMb2_Z7bfPy6XEj6H , then please put this model into subfolder 'DaSiamRPN', so that the tracker can find and load the pretrained_model.

# 结合目标检测和目标跟踪

### 目标检测
目标检测算法使用的是SSD算法，该算法程序基于tensorflow实现。具体见：https://github.com/wcwowwwww/SSD-Tensorflow

论文地址为: https://arxiv.org/abs/1512.02325

预训练的模型可以从 https://drive.google.com/file/d/0B0qPCUZ-3YwWUXh4UHJrd1RDM3c/view?usp=sharing 中下载，下载后放在SSD/checkpoints文件夹中。

### 目标追踪
目标跟踪算法使用的是SiamFPN算法，该算法程序基于pytorch实现。具体见：https://github.com/wcwowwwww/DaSiamRPN

论文地址为: http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf

预训练的模型请在 https://drive.google.com/drive/folders/1BtIkp5pB6aqePQGlMb2_Z7bfPy6XEj6H 下载SiamRPNVOT.model文件后放入DaSiamRPN文件夹中即可。

这两个算法的精度和速度都表现非常好，不需要GPU，能够在CPU条件下流畅运行。（当然训练时需要使用GPU加速运算）

代码执行环境：python, opencv, tensorflow, pytorch均更新为最新即可。

下载代码和模型后运行SSD+SiamRPN.py，可以实现对摄像头前目标的检测和跟踪。

### spaceshoot
spaceshoot文件夹内是一款飞行设计类游戏，运行Detection+Tracking+SpaceShooter.py可以打开游戏，按enter进入游戏后，飞机可以由摄像头前的目标控制移动。


