# _*_ coding: UTF-8 _*_

import matplotlib.pyplot as plt

import numpy as np

#创建数据
x = np.linspace(-2,2,100) #在-2到2之间创建100个点
y = 3*x+4
#创建图像
plt.plot(x,y)
#显示图像
plt.show()

#同时画2个图线
y1 = 3*x+4
y2 = x ** 2
plt.plot(x,y1)
plt.plot(x,y2)
plt.show()