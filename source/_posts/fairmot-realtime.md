---
title: FairMOT实时多目标跟踪系统
top: false
cover: false
toc: true
mathjax: true
reprintPolicy: cc-by
date: 2020-10-24 10:00:00
password:
summary:
tags: FairMOT摄像头实时多目标跟踪
categories: 多目标跟踪
---


## 简介
FairMOT是今年很火的一个多目标跟踪算法，前不久也开放了最新版本的论文，并于最近重构了开源代码，我也在实际工程视频上进行了测试，效果是很不错的。不过，官方源码没有提高实时摄像头跟踪的编程接口，我在源码的基础上进行了修改，增加了实时跟踪模块。本文介绍如何进行环境配置和脚本修改，实现摄像头跟踪（**本文均采用Ubuntu16.04进行环境配置，使用Windows在安装DCN等包的时候会有很多问题，不建议使用**）。

## 环境配置

> 下述环境配置需要保证用户已经安装了git和conda，否则配置pytorch和cuda会诸多不便。

首先，通过下面的git命令从Github克隆源码到本地并进入该项目。访问[链接](https://pan.baidu.com/s/1H1Zp8wrTKDk20_DSPAeEkg)（提取码uouv）下载训练好的模型，在项目根目录下新建`models`目录（和已有的`assets`、`src`等目录同级），将刚刚下载好的模型文件`fairmot_dla34.pth`放到这个`models`目录下。

```shell
git clone git@github.com:ifzhang/FairMOT.git
cd FairMOT
```

下面，通过conda创建适用于该项目的虚拟环境（环境隔离），国内用户速度慢可以参考[我conda的文章](https://zhouchen.blog.csdn.net/article/details/86086919)配置国内源。创建之后通过`activate`激活环境（该命令出错将`conda`换为`source`）。然后在当前虚拟环境下（**后续关于该项目的操作都需要在该虚拟环境下**）安装pytorch和cuda（这里也建议配置国内源后安装`conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch`）。最后，通过pip命令安装所需d的Python包（国内建议[配置清华源](https://zhouchen.blog.csdn.net/article/details/106420275)），注意先安装cython。

```shell
conda create -n fairmot python=3.6
conda activate fairmot
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0
pip install cython
pip install -r requirements.txt
pip install -U opencv-python==4.1.1.26
```

同时，用于项目使用了DCNv2所以需要安装该包，该包只能通过源码安装，依次执行下述命令即可（安装过程报warning是正常情况，不报error就行）。
```shell
git clone https://github.com/CharlesShang/DCNv2
cd DCNv2
./make.sh
cd ../
```

至此，所有的环境配置已经完成，由于这里还需要使用到ffmpeg来生成视频文件，所以系统需要安装ffmpeg（Ubuntu采用apt安装即可），教程很多，不多赘述。

想要试试项目是否正常工作，可以使用下面的命令在demo视频上进行跟踪测试（初次允许需要下载dla34模型，这个模型国内下载速度还可以，我就直接通过允许代码下载的）。

```shell
cd src
python demo.py mot --input-video ../videos/MOT16-03.mp4 --load_model ../models/fairmot_dla34.pth --conf_thres 0.4
```

默认文件输出在项目根目录的`demos`文件夹下，包括每一帧的检测结果以及组合成的视频。

![](https://i.loli.net/2020/10/24/bxVoM7ZyjTBmhLI.png)

![](https://i.loli.net/2020/10/24/NF7idqKkQ8CyfG9.png)

## 实时跟踪

实时跟踪主要在两个方面进行修改，一是数据加载器，二是跟踪器。首先，我们在`src`目录下新建一个类似于`demo.py`的脚本文件名为`camera.py`，写入和`demo.py`类似的内容，不过，我们把视频路径换位摄像机编号（这是考虑到JDE采用opencv进行视频读取，而opencv视频读取和摄像机视频流读取是一个接口）。具体`camera.py`内容如下。

```python
import os

import _init_paths
from opts import opts
from tracking_utils.utils import mkdir_if_missing
import datasets.dataset.jde as datasets
from track import eval_seq


def recogniton():
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)
    print("start tracking")
    dataloader = datasets.LoadVideo(0, opt.img_size)
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate

    frame_dir = None if opt.output_format == 'text' else os.path.join(result_root, 'frame')
    eval_seq(opt, dataloader, 'mot', result_filename,
             save_dir=frame_dir, show_image=False, frame_rate=frame_rate)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    recogniton()
```

接着，原来JDE关于视频加载是针对真正的视频的，对于摄像头这种无限视频流，修改其帧数为无限大（很大很大的整数值即可），也就是将`src/lib/datasets/dataset/jde.py`中`LoadVideo`修改如下。

```python
class LoadVideo:
    def __init__(self, path, img_size=(1088, 608)):
        self.cap = cv2.VideoCapture(path)
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if type(path) == type(0):
            self.vn = 2 ** 32
        else:
            self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        self.w, self.h = 1920, 1080
        print('Lenth of the video: {:d} frames'.format(self.vn))

    def get_size(self, vw, vh, dw, dh):
        wa, ha = float(dw) / vw, float(dh) / vh
        a = min(wa, ha)
        return int(vw * a), int(vh * a)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == len(self):
            raise StopIteration
        # Read image
        res, img0 = self.cap.read()  # BGR
        assert img0 is not None, 'Failed to load frame {:d}'.format(self.count)
        img0 = cv2.resize(img0, (self.w, self.h))

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return self.count, img, img0

    def __len__(self):
        return self.vn  # number of frames
```

至此，读取视频流也通过一个粗暴的方式实现了，然后就是窗口显示了，原来项目中跟踪器只会一帧一帧写入跟踪后的结果图像，然后通过`ffmpeg`将这些图像组合为视频。不过，原项目已经设计了实时显示跟踪结果窗口的接口了，只需要调用`track.py`中的`eval_seq`函数时，参数`show_image`设置为`True`即可。不过，也许作者并没有测试过这个模块，这里显示会有些问题，务必将`eval_seq`中下述代码段进行如下修改。

```python
if show_image:
    cv2.imshow('online_im', online_im)
    cv2.waitKey(1)
```

调整完成后，输入下面的命令运行跟踪脚本（命令行Ctrl+C停止跟踪，跟踪的每一帧存放在指定的`output-root`目录下的`frame`目录中）。

```shell
python camera.py mot --load_model ../models/fairmot_dla34.pth --output-root ../results
```

![](https://i.loli.net/2020/10/24/Kp52w4qlb7nyVRQ.png)

上图是我实际测试得到的运行结果，摄像头分辨率比较低并且我做了一些隐私模糊处理，不过，整个算法的实用性还是非常强的，平均FPS也有18左右（单卡2080Ti）。

## 补充说明
本文对FairMOT源码进行了简单粗暴的修改以实现了一个摄像头视频实时跟踪系统，只是研究FairMOT代码闲暇之余的小demo，具体代码可以在[我的Github](https://github.com/luanshiyinyang/FairMOT)找到。



