# Object-detecting-and-tracking
Combing object detecting and object tracking. The object detecting algorithm is SSD and the object detecting algorithm is SiamRPN. Both algorithms are real-time.

# 目标检测和目标跟踪结合

### 目标检测
目标检测算法使用的是SSD算法，该算法程序基于tensorflow实现。具体见：https://github.com/wcwowwwww/SSD-Tensorflow

论文地址为https://arxiv.org/abs/1512.02325

预训练的模型可以从 https://drive.google.com/file/d/0B0qPCUZ-3YwWUXh4UHJrd1RDM3c/view?usp=sharing 中下载，下载后放在SSD/checkpoints文件夹中。

### 目标追踪
目标跟踪算法使用的是SiamFPN算法，该算法程序基于pytorch实现。具体见：https://github.com/wcwowwwww/DaSiamRPN

论文地址为http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf

预训练的模型请在 https://drive.google.com/drive/folders/1BtIkp5pB6aqePQGlMb2_Z7bfPy6XEj6H 下载SiamRPNVOT.model文件后放入DaSiamRPN文件夹中即可。

这两个算法的精度和速度都表现非常好，不需要GPU，能够在CPU条件下流畅运行。（当然训练时需要使用GPU加速运算）

代码执行环境：python, opencv, tensorflow, pytorch均更新为最新即可。

下载代码后运行SSD+SiamRPN.py，可以实现对摄像头前目标的检测和跟踪。

### spaceshoot
spaceshoot文件夹内是一款飞行设计类游戏，运行Detection+Tracking+SpaceShooter.py可以打开游戏，按enter进入游戏后，飞机可以由摄像头前的目标控制移动。


