---
title: Matplotlib绘制动图
top: false
cover: false
toc: true
mathjax: true
reprintPolicy: cc-by
date: 2020-09-21 16:14:49
password:
summary:
tags: Matplotlib动态图
categories: 数据可视化
---

## 简介
Matplotlib是非常著名的Python绘图库，支持非常复杂的底层定制化操作。本文通过Matplotlib中的动画绘制工具来讲解如何绘制动态图，首先讲解通过交互模式如何显示动态图，继而讲解通过两个动画类来实现动图地保存（GIF格式）。

## 显示动态图
首先，需要明确，Matplotlib绘图有两种显示模式，分别为**阻塞模式**和**交互模式**，他们具体的说明如下。

1. 阻塞模式，该模式下绘制地图地显示必须使用`plt.show()`进行展示（默认会弹出一个窗口），代码会运行到该行阻塞不继续执行，直到关闭这个展示（默认是关闭弹出的显示窗口，Pycharm等集成开发环境会自动捕获图片然后跳过阻塞）。
2. 交互模式，该模式下任何绘图相关的操作如`plt.plot()`会立即显示绘制的图形然后迅速关闭，继续代码的运行，不发生阻塞。

默认情况下，Matplotlib使用阻塞模式，要想打开交互模式需要通过下面的几个函数来做操作，下面直接列出要用的核心函数。

```python
plt.ion()  # 打开交互模式
plt.ioff()  # 关闭交互模式
plt.clf()  # 清除当前的Figure对象
plt.pause()  # 暂停GUI功能多少秒
```

然后就是要清楚，所谓的动图或者视频是怎么做到的，其实它们本质上就是很多静态图以较快的速度连续播放从而给人一种动感，利用Matplotlib绘制动图的原理也是一样的，遵循`画布绘图`->`清理画布`->`画布绘图`的循环就行了，不过这里注意，由于交互模式下绘图都是一闪而过，因此**通过`plt.pause(n)`暂停GUI显示n秒才能得到连续有显示的图像**。

```python
import matplotlib.pyplot as plt
import numpy as np


def io_test():
    fig = plt.figure()  # 生成画布
    plt.ion()  # 打开交互模式
    for index in range(50):
        fig.clf()  # 清空当前Figure对象
        fig.suptitle("3d io pic")

        # 生成数据
        point_count = 100  # 随机生成100个点
        x = np.random.random(point_count)
        y = np.random.random(point_count)
        z = np.random.random(point_count)
        color = np.random.random(point_count)
        scale = np.random.random(point_count) * 100
        ax = fig.add_subplot(111, projection="3d")
        # 绘图
        ax.scatter3D(x, y, z, s=scale, c=color, marker="o")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        # 暂停
        plt.pause(0.2)

    # 关闭交互模式
    plt.ioff()

    plt.show()


if __name__ == '__main__':
    io_test()
```

上述代码演示了三维空间如何动态显示100个随机点的变化，使用录制软件得到的动图如下，其本质就是不停显示不同的图像而已。

![](https://i.loli.net/2020/09/21/lyQHWBXOGv6KSZ3.gif)

## 动图保存
很多时候我们的需求并不是在窗口中动态显示图像，还需要保存到本地GIF图像，显然使用录制工具是一个比较低效的用法，Matplotlib的`animation`模块提供了两个动画绘制接口，分别是**FuncAnimation**和**ArtistAnimation**，它们都是继承自`TimedAnimation`的类，因而也具有`Animation`对象的通用方法，如`Animation.save()`和`Animation.to_html5_video()`两个方法实例化一个`Animation`对象后均可调用，前者表示将动画保存为一个图像，后者表示将动画表示为一个HTML视频。

- FuncAnimation: 通过反复调用同一更新函数来制作动画。
- ArtistAnimation: 通过调用一个固定的Artist对象来制作动画，例如给定的图片序列或者Matplotlib的绘图对象。

下面给出上述两个类的构造函数所需参数，它们的主要参数也是类似的，都是一个Figure对象作为画布，然后一个对象作为更新的实现方式（前者需要一个反复绘图的更新函数，后者则为一个图像列表或者绘图对象列表）。

```python
ani = animation.FuncAnimation(fig, func, frames=None, init_func=None, fargs=None, save_count=None, *, cache_frame_data=True, **kwargs)
ani = animation.ArtistAnimation(fig, artists, *args, **kwargs)
```

相比较而言，我更喜欢使用`FuncAnimation`，它的使用要求简洁且定制化程度较高。但是如果想将很多图片合并为一个动图，那么`ArtistAnimation`是最合适的选择。

下面的代码演示了如何保存一个动态变化渲染的柱状图，`ArtistAnimation`传入了一个图像序列，序列中每个元素为绘制的柱状图。然后通过使用`Animation`的`save`方法保存了动态图，**需要注意的是，这里有个动画写入器（writer）可以选择，默认不是pillow，我个人觉得pillow安装简单一些。

```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()
x, y, tmp = [], [], []

for i in range(10):
    x.append(i)
    y.append(i+1)
    temp = ax.bar(x, height=y, width=0.3)
    tmp.append(temp)

ani = animation.ArtistAnimation(fig, tmp, interval=200, repeat_delay=1000)
ani.save("bar.gif", writer='pillow')
```

上面代码的执行结果如下图。

![](https://i.loli.net/2020/09/21/LHoTe14VU2tqlX9.gif)

接着，演示使用范围更广的`FuncAnimation`如何使用。下面的代码中，动态展示了梯度下降在三维图上的优化过程，其中最为核心的代码如下。用于构造`Animation`对象的除了画布就是一个更新函数，在这个更新函数内部多次绘制散点图从而形成动态效果， `frames`是帧数，如果设置了这个帧数，那么`update`函数第一个参数必须有一个`num`占位，这个`num`由`Animation`对象维护，每次内部执行`update`会自动递增，后面的参数列表`fargs`只需要传入除了`num`后面的参数即可。

```python
def update(num, x, y, z, ax):
    x, y, z = x[:num], y[:num], z[:num]
    ax.scatter3D(x, y, z, color='black', s=100)
    return ax

ani = animation.FuncAnimation(fig, update, frames=25, fargs=(x_list, y_list, z_list, ax3d), interval=50, blit=False)
```

上面的代码演示效果如下图，完整的代码附在文末补充说明中。

![](https://i.loli.net/2020/09/21/Kp6qMkuLIt3rFwo.gif)

## 补充说明
本文介绍了如何使用Matplotlib绘制动态图，主要通过交互模式和`animation`模块进行，如果觉得有所帮助，欢迎点赞。

```python
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


def GD(x0, y0, lr, epoch):
    f = lambda x, y: x ** 2 - y ** 2
    g_x = lambda x: 2 * x
    x, y = x0, y0
    x_list, y_list, z_list = [], [], []
    for i in range(epoch):
        x_list.append(x)
        y_list.append(y)
        z_list.append(f(x, y) * 1.01)

        grad_x, grad_y = g_x(x), g_x(y)
        x -= lr * grad_x
        y -= lr * grad_y
        print("Epoch{}: grad={} {}, x={}".format(i, grad_x, grad_y, x))
        if abs(grad_x) < 1e-6 and abs(grad_y) < 1e-6:
            break
    return x_list, y_list, z_list


def update(num, x, y, z, ax):
    x, y, z = x[:num], y[:num], z[:num]
    ax.scatter3D(x, y, z, color='black', s=100)
    return ax


def draw_gd():
    fig = plt.figure()
    x, y = np.meshgrid(np.linspace(-3, 3, 1000), np.linspace(-3, 3, 1000))
    z = x ** 2 - y ** 2
    ax3d = plt.gca(projection='3d')
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    plt.tick_params(labelsize=10)
    ax3d.plot_surface(x, y, z, cstride=20, rstride=20, cmap="jet")

    x_list, y_list, z_list = GD(-3, 0, 0.01, 100)
    x_list, y_list, z_list = np.array(x_list), np.array(y_list), np.array(z_list)

    ani = animation.FuncAnimation(fig, update, frames=25, fargs=(x_list, y_list, z_list, ax3d), interval=50, blit=False)
    ani.save('test.gif')


if __name__ == '__main__':
    draw_gd()
```




