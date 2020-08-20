---
title: anchor-free detection
top: false
cover: false
toc: true
mathjax: true
reprintPolicy: cc-by
date: 2020-08-20 10:01:11
password:
summary:
tags: anchor-free目标检测
categories: 目标检测
---

# Anchor-free目标检测

## 简介
沿着two-step到one-step，anchor-based到anchor-free的发展路线，如今，目标检测（Object Detection，OD）已经进入anchor-free的时代。这些anchor-free方法因为适用于多目标跟踪等领域，也促使了MOT的飞速发展。本文将沿着anchor-free目标检测发展地路线，简单介绍一下主要地一些方法思路，包括目前关注度较大地**FCOS**和**CenterNet**。


## 何为anchor
在了解anchor-free的方法前，我们得知道什么是anchor，在过去，目标检测通常被建模为对候选框的分类和回归，不过，按照候选区域的产生方式不同，分为二阶段（two-step）检测和单阶段（one-step）检测，前者的候选框通过RPN（区域推荐网络）网络产生proposal，后者通过滑窗产生anchor。

![](https://i.loli.net/2020/08/20/keLHXrlPFBnqjAf.png)

本文所提到的anchor-free方法则通过完全不同的思路解决目标检测问题，这些思路都没有采用预定义的候选框的概念。这两年，从CornerNet开始，anchor-free的目标检测框架层出不穷，宣告着目标检测迈入anchor-free时代。

## anchor-free发展
其实anchor-free不是一个很新的概念，最早可以追溯到YOLO算法，这应该是最早的anchor-free模型，而最近的anchor-free方法主要分为**基于密集预测**和**基于关键点估计**两种。

### 早期研究
先是聊一聊目标检测比较古老的研究，分别是Densebox和YOLO，前者发表于2015年9月，后者则开源于2016年。

**Densebox**

首先来聊一聊Densebox，这是地平线的算法工程师黄李超于2015年设计的一个端到端检测框架，对此有专门的[文章](https://zhuanlan.zhihu.com/p/24350950)介绍。Densebox是深度学习用于目标检测的开山之作之一，当时已经有不错效果的R-CNN不够直接且高效，因而Densebox作者从OverFeat方法上得到启发：在图像上进行卷积等同于使用滑窗分类，为何不能使用全卷积对整个图像进行目标检测呢？

![](https://i.loli.net/2020/08/20/fMqbOVpkK1rFSsE.png)

在这个基础上，设计了一套端到端的多任务全卷积模型，如上图所示。该模型可以直接回归出目标出现的置信度和相对位置，同时为了处理遮挡和小目标，引入上采用层融合浅层网络特征，得到更大的尺寸的输出特征图。下图是网络的输入和输出，对每个像素会得到一个5维向量，表示分类置信度和bbox到该pixel的四个距离。

![](https://i.loli.net/2020/08/20/sZfUd1VOzECyGXv.png)

Densebox的主要贡献有两个：证明了单FCN（全卷积网络）可以实现检测遮挡和不同尺度的目标；在FCN结构中添加少量层引入landmark localization，将landmark heatmap和score map融合能够进一步提高检测性能。

**遗憾的是，当时目标检测的学者们沿着RCNN铺好的路亦步亦趋，现在想想，如果当时，就有足够多的关注给与Densebox，今天的目标检测是否会是全新的局面呢？**

**YOLO**

2016年开源的YOLOv1算法，是目前工业界比较关注的算法之一，它开创性地将目标检测中的候选框生成和目标识别通过一步完成，因而论文名为“You only look once”，YOLO模型可以直接从整个图像上得到边界框和对应的置信度。比较详细的理解可以参考[我之前YOLO算法的文章](https://zhouchen.blog.csdn.net/article/details/105178437)。

![](https://i.loli.net/2020/08/20/whpbPxLcqWmsytH.png)

YOLO的最大创新就是速度快，一方面将候选框生成的步骤去除，另一方面，通过多个网格负责目标的检测，大大加快运行速度。

> Densebox和YOLO很类似，都可以理解为单阶段检测，不过前者为密集预测，针对每个像素进行预测；后者针对划分得到的网格进行预测。同时，作为anchor-free的两篇开山之作，它们为后来的anchor-free检测提供了很多的思路。

### 基于密集预测
沿着上一节YOLO和Densebox的思路，2019年出现了很多以此为基础的目标检测方法，包括FCOS、FSAF以及FoveaBox等等方法。

**FCOS**

这是这两年受到广泛的关注的目标检测算法，一方面它确实是anchor-free系列打破anchor-based精度神话的关键之作，另一方面，业界对这种单阶段高效算法有着巨大的需求。

![](https://i.loli.net/2020/08/20/UVEgZsIvuQtBPTm.png)

上图是FCOS的pipeline设计图，核心的就是一个特征金字塔和三分支头网络，通过backbone之后对feature map上每一个点进行回归预测，和以往目标检测任务不同的是，**除了分类和回归分支，加入了center-ness以剔除低质量预测，它和分类分支的乘积为最终的置信度。**

FCOS创新点如下:
1. 突破基于Faster-RCNN修补的思路，开创性地不使用anchor而是直接对每个像素进行预测，并在效果是远超Faster-RCNN。这主要是考虑到anchor地存在会带来大量地超参数，如anchor number等，而且这些anchor要计算和GT地IOU，也是很消耗算力的。
2. 由于是像素级别的密集预测，因此可以使用分割任务的一些trick并且通过修改目标分支可用于实例分割和关键点检测等任务。
3. 由于是全卷积网络，拥有很多FCN任务的优势，也可以借用其思想。

**FSAF**

![](https://i.loli.net/2020/08/20/j6QLvatzU9DmlBn.png)

这是一个针对FPN的优化思路，提出FSAF模块，让网络自己学习anchor适配。​在RetinaNet的基础上，FSAF模块引入了2个额外的卷积层，这两个卷积层各自负责anchor-free分支的分类和回归预测。此外，提出了在线特征选择策略，​实例输入到特征金字塔的所有层，然后求得所有anchor-free分支focal loss和IoU loss的和，选择loss最小的特征层来学习实例。训练时，特征根据安排的实例进行更新。推理时，不需要进行特征更新，因为最合适的特征金字塔层必然输出高置信分数。

> 虽然都是基于密集预测，但相比于YOLO和Densebox，FCOS和FSAF使用FPN进行多尺度预测，此前的方法只有单尺度预测；不过，相比于YOLO这个单分支模型，其他方法都是通过两个子网络来进行分类和回归。

### 基于关键点估计
不同于密集预测的思路，以关键点估计为手段，目标检测出现了一条全新的主线，它彻底抛开了区域分类回归思想，主要出现了CornerNet、ExtremeNet以及集大成者的CenterNet，由于有两篇目标检测的文章网络名都是CenterNet，这里特指的是关注度比较高的Objects as points这篇文章。

**CornerNet**

这篇文章是后来很多基于关键点估计处理目标检测的算法基础，它开创性地用一对角点来检测目标。对一幅图像，预测两组heatmap，一组为top-left角点，另一组为bottom-right角点，每组heatmap有类别个通道。下图为框架图。

![](https://i.loli.net/2020/08/20/1SjiLUVbJGzXr9o.png)

**ExtremeNet**

不同于CornerNet使用角点检测目标，ExtremeNet通过极值点和中心点来检测目标，这应该是最大地区别，其他一些关键点估计方面地细节，这里不多提。

![](https://i.loli.net/2020/08/20/QuToZCtJUprdxwX.png)


**CenterNet**

下面来看看关键点估计用于目标检测地集大成者，CenterNet。抛开了传统的边框目标表示方法，将目标检测视为对一个点进行的关键点估计问题。相比较于基于bbox的方法，该模型端到端可微，其简单高效且实时性高。在主流地OD数据集上超越了大部分SOTA方法，且论文称在速度上超越了YOLO3。

![](https://i.loli.net/2020/08/20/CIp6humX5QG9dPN.png)

通过中心点来表示目标，然后在中心点位置回归出目标的其他属性，这样，目标检测问题变成了一个关键点估计问题。只需要将图像传入全卷积网络，得到热力图，热力图的峰值点就是中心点。这里可以把中心点看做形状未知的锚点。但是该锚点只在位置上，没有尺寸框，没有阈值进行前后景分类；每个目标只会有一个正的锚点，因此不会用到NMS；而且，CenterNet与传统目标检测相比，下采样倍数较低，不需要多重特征图。

## 发展思路
### 成功原因
anchor-free能在精度上追赶上anchor-based方法，最大地原因应该归属上面绝大多数方法避不开地FPN（特征金字塔网络），因为在每个位置只预测一个框地前提下，FPN对尺度信息进行了很好地弥补，而Focal loss则对区域地回归有一定辅助效果。
### anchor-free局限性
当然，anchor-free地目标检测方法也有很大地局限性，这些方法虽然声称精度追上了较好地二阶段方法，但存在一些训练上地细节以及部分不公平地比较。不过，总体来说，速度上地突破还是吸引了很多工业界的关注的。
### GT的设计
上面的很多方法其实出发点都是bbox这个矩形框冗余信息太多，目标信息少，大部分是背景。它们大多都改变了GT的定义，如CornerNet将其定义为角点，ExtremeNet将其定义为极值点，FCOS虽然还是矩形框但也使用了center-ness进行抑制，FSAF则将GT定义为中心区域。对于GT目标的改进优化促使了目标检测的发展。


## 参考文献
**Densebox**: Huang L, Yang Y, Deng Y, et al. Densebox: Unifying landmark localization with end to end object detection[J]. arXiv preprint arXiv:1509.04874, 2015.<br>
**YOLO**: Redmon J, Divvala S, Girshick R, et al. You only look once: Unified, real-time object detection[A]. Proceedings of the IEEE conference on computer vision and pattern recognition[C]. 2016: 779–788.<br>
**FCOS**: Tian Z, Shen C, Chen H, et al. FCOS: Fully Convolutional One-Stage Object Detection[J]. arXiv:1904.01355 [cs], 2019.<br>
**FSAF**: Zhu C, He Y, Savvides M. Feature Selective Anchor-Free Module for Single-Shot Object Detection[J]. arXiv:1903.00621 [cs], 2019.<br>
**CornerNet**: Law H, Deng J. CornerNet: Detecting Objects as Paired Keypoints[J]. arXiv:1808.01244 [cs], 2019.<br>
**ExtremeNet**: Zhou X, Zhuo J, Krähenbühl P. Bottom-up Object Detection by Grouping Extreme and Center Points[J]. arXiv:1901.08043 [cs], 2019.<br>
**CenterNet**: Zhou X, Wang D, Krähenbühl P. Objects as points[A]. arXiv preprint arXiv:1904.07850[C]. 2019.








