import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([3, 5, 7, 6, 2, 6, 10, 15])
plt.plot(x, y, 'r')  # 折线 1x 2y 3color
plt.plot(x, y, 'g', lw=10)  # 4 line width
# 折线 饼状 柱状
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([13, 25, 17, 36, 21, 16, 10, 15])
plt.bar(x, y, 0.5, alpha=1, color='b')  # alpha 透明度  0.5柱状图的占比宽度
plt.show()
