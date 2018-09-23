# _*_ coding: UTF-8 _*_

import matplotlib.pyplot as plt

import numpy as np

#创建数据
x = np.linspace(-4,4,50) #在-4到4之间创建50个点
#同时画2个图线
y1 = 3*x+4
y2 = x ** 2
#构建第一张图
plt.figure(num=1,figsize=(7,6))
plt.plot(x,y1)
#颜色为红色，线宽为3，线为虚线
plt.plot(x,y2,color="red",linewidth=3.0,linestyle="--")

#构建第二张图
plt.figure(num=2)
plt.plot(x,y2,color="green")
plt.show()