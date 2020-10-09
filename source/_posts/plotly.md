---
title: Plotly基础教程
top: false
cover: false
toc: true
mathjax: true
reprintPolicy: cc-by
date: 2020-10-09 11:56:33
password:
summary:
tags: Plotly教程
categories: 数据可视化
---


## 简介

Plotly 是一个非常强大的开源数据可视化框架，它通过构建基于 HTML 的交互式图表来显示信息，可创建各种形式的精美图表。本文所说的 Plotly 指的是 Plotly.js 的 Python 封装，[plotly](https://plotly.com/)本身是个生态非常复杂的绘图工具，它对很多编程语言提供接口。交互式和美观易用应该是 Plotly 最大的优势，而 Matplotlib 的特点则是可定制化程度高，但语法也相对难学，各有优缺点。

## 安装及开发工具

安装通过 PIP 进行即可。

`pip install plotly`

Plotly Python 其对应的官网为https://plotly.com/python/，上面有一些教程和官方API接口的查询。

![](https://i.loli.net/2020/10/09/BHfGe7nIWZ49oVt.png)

**上面说了 Plotly 是基于 HTML 显示的，所以这里推荐使用 Jupyter lab（Jupyter notebook 也行）作为开发工具，Jupyter lab 的安装本文不多提及，可以自行查找。尤其注意的是，Plotly 主要维护 Jupyter notebook，所以对 Jupyter lab 支持不是很好，绘图无法显示，最新版 Plotly 需要通过命令`conda install nodejs`和`jupyter labextension install jupyterlab-plotly@4.11.0`安装支持插件。**

## Plotly 生态

- Plotly 是绘图基础库，它可以深度定制调整绘图，但是 API 复杂学习成本较高。
- Plotly_exprress 则是对 Plotly 的高级封装，上手容易，它对 Plotly 的常用绘图函数进行了封装。缺点是没有 plotly 那样自由度高，个人感觉类似 Seaborn 和 Matplotlib 的关系。**本文不以express为主。**
- Dash 用于创建交互式绘图工具，可以方便地用它来探索数据，其绘图基于 Plotly。使用 Dash 需要注册并购买套餐，也就是常说的“在线模式”，一般，我们在 Jupyter 内本地绘图就够用了，这是“离线模式”。

## 绘图教程

下面涉及到的内容均可以在[官方文档](https://plotly.com/python/)找到参考，下面的内容也只涉及基础的图形绘制（使用Plotly实现），一些比较基础的图形库知识查看[对应教程](https://plotly.com/python/plotly-fundamentals/)。

### 基本图表

在 Plotly 中，预定义了如下的一些基本图表，包括散点图、折线图、柱状图、饼图等，它们的使用方式都是类似的，通过向Figure上添加绘图对象进行绘图，而向绘图对象传递的就是其需要的格式的数据。

![](https://i.loli.net/2020/10/09/HDkcCSLxO1Q83eo.png)

```python
import plotly.graph_objects as go
import numpy as np

N = 1000
t = np.linspace(0, 10, 100)
y = np.sin(t)

fig = go.Figure(data=go.Scatter(x=t, y=y, mode='markers'))

fig.show()
```

下面的代码就是简单散点图的绘制，其在Jupyter中的执行结果如下图，不妨来仔细看看这张图，**在绘制的图形右上角有一行菜单栏且这个图形是可交互的**（包括缩放、旋转、裁剪等），右上角的菜单包括图像下载、缩放、裁剪、在dash中编辑等。

![](https://i.loli.net/2020/10/09/VrR52bqpQA4yzea.png)

也可以通过`add_trace`来逐个添加绘图对象

```python
import plotly.graph_objects as go

# Create random data with numpy
import numpy as np
np.random.seed(1)

N = 100
random_x = np.linspace(0, 1, N)
random_y0 = np.random.randn(N) + 5
random_y1 = np.random.randn(N)
random_y2 = np.random.randn(N) - 5

fig = go.Figure()
# Add traces
fig.add_trace(go.Scatter(x=random_x, y=random_y0, mode='markers', name='markers'))
fig.add_trace(go.Scatter(x=random_x, y=random_y1, mode='lines+markers', name='lines+markers'))
fig.add_trace(go.Scatter(x=random_x, y=random_y2, mode='lines', name='lines'))
fig.show()
```

![](https://i.loli.net/2020/10/09/2n6AJes5rNKqtiH.png)

其他基本图表类似，传入指定格式的数据即可。

### 统计图表

很多统计学图表也预先定义在了Plotly中，主要包括下图所示的箱型图、直方图、热力图、等高线图等。

![](https://i.loli.net/2020/10/09/GpthdgN5PvaOEyz.png)

和上面的基本图表类似，绘图方式是固定的，只是绘图对象改变了而已。下面的代码就是直方图绘制

```python
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Bar(
    name='Control',
    x=['Trial 1', 'Trial 2', 'Trial 3'], y=[3, 6, 4],
    error_y=dict(type='data', array=[1, 0.5, 1.5])
))
fig.add_trace(go.Bar(
    name='Experimental',
    x=['Trial 1', 'Trial 2', 'Trial 3'], y=[4, 7, 3],
    error_y=dict(type='data', array=[0.5, 1, 2])
))
fig.update_layout(barmode='group')
fig.show()
```

![](https://i.loli.net/2020/10/09/rwXYOZnCmokGjaI.png)

### AI图表

同时，Plotly也支持绘制一些简单的机器学习图表，不过都是依靠上面的基本图表实现的，如下述的线性回归。

![](https://i.loli.net/2020/10/09/UIqydzwFvbRK5Q4.png)

```python
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

df = px.data.tips()
X = df.total_bill.values.reshape(-1, 1)

model = LinearRegression()
model.fit(X, df.tip)

x_range = np.linspace(X.min(), X.max(), 100)
y_range = model.predict(x_range.reshape(-1, 1))

fig = px.scatter(df, x='total_bill', y='tip', opacity=0.65)
fig.add_traces(go.Scatter(x=x_range, y=y_range, name='Regression Fit'))
fig.show()
```

![](https://i.loli.net/2020/10/09/Q12kFUBOcmKgqX5.png)

### 科学绘图

下面的这些数据科学领域用的挺多的图也做了封装，例如下面的代码就是绘制heatmap的样例。

![](https://i.loli.net/2020/10/09/2tSgERp8NkrxJDi.png)

```python
import plotly.graph_objects as go

fig = go.Figure(data=go.Heatmap(
                   z=[[1, None, 30, 50, 1], [20, 1, 60, 80, 30], [30, 60, 1, -10, 20]],
                   x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                   y=['Morning', 'Afternoon', 'Evening'],
                   hoverongaps = False))
fig.show()
```

![](https://i.loli.net/2020/10/09/ElyNPz96JOrmnuF.png)

### 三维绘图

Plotly的三维绘图真的很好看，而且其是可交互的，非常方便，例如下面的3D曲面图。

![](https://i.loli.net/2020/10/09/aP16hAGqZrT4C8p.png)

```python
import plotly.graph_objects as go

import pandas as pd

# Read data from a csv
z_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')

fig = go.Figure(data=[go.Surface(z=z_data.values)])

fig.update_layout(title='test', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))

fig.show()
```

![](https://i.loli.net/2020/10/09/DT2Oz5plMsvkCFJ.png)

## 补充说明

本文简单介绍了Plotly的基本绘图方式，其实只要了解了Plotly的生态，使用它并不难，更多的子图、标注等技巧本文没有涉及，还是建议到[官网教程](https://plotly.com/python/)查看，非常易读。



