---
title: yolo5-train
top: true
cover: true
toc: true
mathjax: true
reprintPolicy: cc-by
date: 2020-08-20 17:44:33
password:
summary:
tags: yolov5自定义数据集训练
categories: 目标检测
---
# YOLOv5自定义数据集训练

## 简介
本文介绍如何在自己的VOC格式数据集上训练YOLO5目标检测模型。

## VOC数据集格式
首先，先来了解一下[Pascal VOC数据集](http://host.robots.ox.ac.uk/pascal/VOC/)的格式，该数据集油5个部分组成，文件组织结构如下，目前主要的是VOC2007和VOC2012.

```
- VOC
    - JPEGImages
        - 1.jpg
        - 2.jpg
        - ...
    - Annotations
        - 1.xml
        - 2.xml
        - ...
    - ImageSets
        - Main
            - train.txt
            - val.txt
            - test.txt
            - trainval.txt
        - ...
    - SegmentationClass
    - SegmentationObject
```
第一个文件夹**JPEGImages**为所有的图像，也就是说，训练集、验证集和测试集需要自己划分；**Annotations**为JPEGImages文件夹中每个图片对应的标注，xml格式文件，文件名与对应图像相同；**ImageSets**主要的子文件夹为Main，其中有四个文本文件，为训练集、验证集、测试集和训练验证集的图片文件名；**SegmentationClass**和**SegmentationObject**文件夹存放分割的结果图，前者为语义分割，后者为实例分割。

上述xml标注文件，格式如下。对其具体标注解释。
```xml
<annotation>
  <folder>down</folder> # 图片所处文件夹
  <filename>1.jpg</filename> # 图片文件名及后缀
  <path>./savePicture/train_29635.jpg</path> # 存放路径
  <source>  #图源信息
    <database>Unknown</database>  
  </source>
  <size> # 图片尺寸和通道
    <width>640</width>
    <height>480</height>
    <depth>3</depth>
  </size>
  <segmented>0</segmented>  #是否有分割label，0无1有
  # 图像中包含的所有目标，一个目标一个object标签
  <object>
    <name>car</name>  # 目标类别
    <pose>Unspecified</pose>  # 目标的姿态
    <truncated>0</truncated>  # 目标是否被部分遮挡（>15%）
    <difficult>0</difficult>  # 是否为难以辨识的目标， 需要结合背景才能判断出类别的物体
    <bndbox>  # 目标边界框信息
      <xmin>2</xmin>
      <ymin>156</ymin>
      <xmax>111</xmax>
      <ymax>259</ymax>
    </bndbox>
  </object>
  <object>
      <name>multi_signs</name>
      <editType />
      <pose>Unspecified</pose>
      <truncated>0</truncated>
      <difficult>0</difficult>
      <bndbox>
         <xmin>81</xmin>
         <ymin>98</ymin>
         <xmax>154</xmax>
         <ymax>243</ymax>
      </bndbox>
   </object>
</annotation>
```

**也就是说，遇到这种文件格式的数据（主要特点为图像全放在一个文件夹，标注格式如上等），将其作为VOC格式的数据集，将自己的数据集重构为VOC格式以便开源项目的处理。**

## 自定义训练

### **下载源码**
通过`git clone git@github.com:ultralytics/yolov5.git`将YOLOv5源码下载到本地，本文后面的内容也可以参考[官方的自定义数据集训练教程](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)，不同于我的教程，该教程全面包含了VOC格式和COCO格式数据集的处理方法。

此时创建虚拟环境，并通过`pip install -r requirements.txt`安装依赖包，我这里测试过，最新的项目是兼容Pytorch 1.6的，1.6之前的Pytorch会有一些问题。

### **数据集处理**
一般，符合VOC格式的数据集至少包含图像和标注两个文件夹，结构如下。我这里假定测试集是独立的，该数据集实际上为训练集，只需要划分出训练集和验证集即可。**这里建议将文件夹重命名如下，否则后续可能出现数据集加载失败的情况。**

```
- 根目录
    - images
    - Annotations
```

下面，编写脚本划分数据集，`split_train_val.py`脚本内容如下（参考Github上的开源脚本），只需要执行`python split_train_val.py --xml_path dataset_root/Annotations/ --txt_path dataset_root/anno_txt/`就得到了划分结果的文件列表，如训练集对应的`train.txt`如下图，里面与训练图片所有的文件名。

```python
import os
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--xml_path', type=str, help='input xml label path')
parser.add_argument('--txt_path', type=str, help='output txt label path')
opt = parser.parse_args()

trainval_percent = 1.0
train_percent = 0.8
xmlfilepath = opt.xml_path
txtsavepath = opt.txt_path
total_xml = os.listdir(xmlfilepath)
if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)

num = len(total_xml)
list_index = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list_index, tv)
train = random.sample(trainval, tr)

file_trainval = open(txtsavepath + '/trainval.txt', 'w')
file_test = open(txtsavepath + '/test.txt', 'w')
file_train = open(txtsavepath + '/train.txt', 'w')
file_val = open(txtsavepath + '/val.txt', 'w')

for i in list_index:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        file_trainval.write(name)
        if i in train:
            file_train.write(name)
        else:
            file_val.write(name)
    else:
        file_test.write(name)

file_trainval.close()
file_train.close()
file_val.close()
file_test.close()
```

![](https://i.loli.net/2020/08/20/HY5yUEdALgmVthw.png)

接下来，我们要做的就是每个xml标注提取bbox信息为txt格式，每个图像对应一个txt文件，文件每一行为一个目标的信息，包括`类别 xmin xmax ymin ymax`。使用的脚本`voc_label.py`内容如下（**注意，类别要替换为当前数据集的类别列表**），**在数据集根目录（此时包含Annotations、anno_txt以及images三个文件夹的目录）下执行该脚本**，如`python ../../utils/voc_label.py`。

```python
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import os
from os import getcwd

sets = ['train', 'val', 'test']
classes = ['window_shielding', 'multi_signs', 'non_traffic_signs']
abs_path = os.getcwd()


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(image_id):
    in_file = open('Annotations/%s.xml' % (image_id))
    out_file = open('labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


wd = getcwd()
for image_set in sets:
    if not os.path.exists('labels/'):
        os.makedirs('labels/')
    image_ids = open('anno_txt/%s.txt' % (image_set)).read().strip().split()
    list_file = open('%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        list_file.write(abs_path + '/images/%s.jpg\n' % (image_id))
        convert_annotation(image_id)
    list_file.close()
```
这时候，我们的目标检测数据集就构建完成了，其内容如下，其中labels中为不同图像的标注文件，`train.txt`等几个根目录下的txt文件为划分后图像所在位置的绝对路径，如`train.txt`就含有所有训练集图像的绝对路径。

![](https://i.loli.net/2020/08/20/NPwkyM3o4T2cn9m.png)

### 配置文件
下面需要两个配置文件用于模型的训练，一个用于数据集的配置，一个用于模型的配置。

首先是数据集的配置，在根目录下的data目录下新建一个yaml文件，内容如下，首先是训练集和验证集的划分文件，这个文件在上面一节最后生成得到了，然后是目标的类别数目和具体类别列表，这个列表务必和上一节最后`voc_label.py`中的一致。

```yaml
train: dataset/train.txt
val: dataset/val.txt
 
# number of classes
nc: 3
 
# class names
names: ['window_shielding', 'multi_signs', 'non_traffic_signs']
```

然后，编辑模型的配置文件，此时需要先在项目根目录下的weights目录下执行其中的download_weights.sh这个shell脚本来下载四种模型的权重。然后，选择一个模型，编辑项目根目录下models目录中选择的模型的配置文件，将第一个参数nc改为自己的数据集类别数即可，例如我使用yolov5x模型，则修改yolov5x.yaml文件。**这里weights的下载可能因为网络而难以进行，我也将其上传到了百度网盘，[地址](链接：https://pan.baidu.com/s/1UQX6URxaJP0ZqALvWpDWkA)给出，提取码为vjlx。**

### 模型训练

此时，可以使用下面的命令进行模型的训练，训练日志默认保存在`./runs/`下，包括模型参数、Tensorboard记录等。此时TensorBoard以已经默认打开，浏览器访问效果如下图（由于数据量很小，很快过拟合）。

```shell
python train.py --img 640 --batch 8 --epoch 300 --data ./data/ads.yaml --cfg ./models/yolov5x.yaml --weights weights/yolov5x.pt --device '0'
```

![](https://i.loli.net/2020/08/20/GrLI9OTtZD3fJFH.png)

### 模型测试
最后，测试模型，使用下面的命令（该命令中`save-txt`选项用于生成结果的txt标注文件，不指定则只会生成结果图像）。其中，weights使用最满意的实验即可，source则提供一个包含所有测试图片的文件夹即可。

```shell
 python detect.py --weights runs/exp0/weights/best.pt --source ./dataset/test/ --device 0 --save-txt
```

这样，对每个测试图片会在默认的`inference/output`文件夹中生成一个同名的txt文件，按照我的需求修改了`detect.py`文件后，每个txt会生成一行一个目标的信息，信息包括`类别序号 置信度 xcenter ycenter w h`，后面四个为bbox位置，均未归一化。如下图。

![](https://i.loli.net/2020/08/20/l86zj2dw9xHnTFO.png)

我这里因为是一个比赛，再将这个txt处理为了json文件。**不论是这里的处理代码还是上面对`detec.py`修改的代码，都可以在文末给出的Github仓库找到。**

## 补充说明
本文介绍了如何使用YOLOv5在自己的数据集上进行训练，按部就班地进行了讲解。该项目在YOLOv5地源码基础上修改完成，代码开源于我的Github，欢迎star或者fork。
 


