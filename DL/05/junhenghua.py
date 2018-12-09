# 灰度 直方图均衡化
"""
import cv2
import numpy as np

img = cv2.imread("../image0.jpg", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("src", img)
dst = cv2.equalizeHist(gray)
cv2.imshow("dst", dst)
cv2.waitKey(0)
"""

# 彩色 直方图均衡化
"""
import cv2
import numpy as np

img = cv2.imread("../image0.jpg", 1)
cv2.imshow("src", img)
(b, g, r) = cv2.split(img)  # 通道分解
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
result = cv2.merge((bH, gH, rH))  # 通道合成
cv2.imshow("dst", result)
cv2.waitKey(0)
"""

# YUV 直方图均衡化
"""
import cv2
import numpy as np

img = cv2.imread("../image0.jpg", 1)
imgYUV = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
cv2.imshow("src", img)
channelYUV = cv2.split(imgYUV)
channelYUV[0] = cv2.equalizeHist(channelYUV[0])
channels = cv2.merge(channelYUV)
result = cv2.cvtColor(channels, cv2.COLOR_YCrCb2BGR)
cv2.imshow("dst", result)
cv2.waitKey(0)
"""

# 1 0-255 2 概率
# 本质：统计每个像素灰度 出现的概率 0-255 p
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../image0.jpg", 1)
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
count = np.zeros(256, np.float)
for i in range(0, height):
    for j in range(0, width):
        pixel = gray[i, j]
        index = int(pixel)
        count[index] = count[index] + 1
for i in range(0, 255):
    count[i] = count[i] / (height * width)
x = np.linspace(0, 255, 256)
y = count
plt.bar(x, y, 0.9, alpha=1, color='b')
plt.show()
cv2.waitKey(0)
"""

# 本质：统计每个像素灰度 出现的概率 0-255 p
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../image0.jpg', 1)
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]

count_b = np.zeros(256, np.float)
count_g = np.zeros(256, np.float)
count_r = np.zeros(256, np.float)
for i in range(0, height):
    for j in range(0, width):
        (b, g, r) = img[i, j]
        index_b = int(b)
        index_g = int(g)
        index_r = int(r)
        count_b[index_b] = count_b[index_b] + 1
        count_g[index_g] = count_g[index_g] + 1
        count_r[index_r] = count_b[index_r] + 1
for i in range(0, 256):
    count_b[i] = count_b[i] / (height * width)
    count_g[i] = count_g[i] / (height * width)
    count_r[i] = count_r[i] / (height * width)
x = np.linspace(0, 255, 256)
y1 = count_b
plt.figure()
plt.bar(x, y1, 0.9, alpha=1, color='b')
y2 = count_g
plt.figure()
plt.bar(x, y2, 0.9, alpha=1, color='g')
y3 = count_r
plt.figure()
plt.bar(x, y3, 0.9, alpha=1, color='r')
plt.show()
cv2.waitKey(0)
"""

"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../image0.jpg", 1)
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
count = np.zeros(256, np.float)
for i in range(0, height):
    for j in range(0, width):
        pixel = gray[i, j]
        index = int(pixel)
        count[index] = count[index] + 1
for i in range(0, 255):
    count[i] = count[i] / (height * width)
# 计算累计概率
sum1 = float(0)
for i in range(0, 256):
    sum1 = sum1 + count[i]
    count[i] = sum1
print(count)
# 计算映射表
map1 = np.zeros(256, np.uint16)
for i in range(0, 256):
    map1[i] = np.uint16(count[i] * 256)
# 映射
for i in range(0, height):
    for j in range(0, width):
        pixel = gray[i, j]
        gray[i, j] = map1[pixel]
cv2.imshow("dst", gray)
cv2.waitKey(0)
"""

# p = p +40
"""
import cv2
import numpy as np

img = cv2.imread("../image0.jpg", 1)
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
cv2.imshow('src', img)
dst = np.zeros((height, width, 3), np.uint8)
for i in range(0, height):
    for j in range(0, width):
        (b, g, r) = img[i, j]
        bb = int(b) + 40
        gg = int(g) + 40
        rr = int(r) + 40
        if bb > 255:
            bb = 255
        if gg > 255:
            gg = 255
        if rr > 255:
            rr = 255
        dst[i, j] = (bb, gg, rr)
cv2.imshow("dst", dst)
cv2.waitKey(0)
"""

# 双边滤波
import cv2

img = cv2.imread("../image0.jpg", 1)
cv2.imshow('src', img)
dst = cv2.bilateralFilter(img, 15, 35, 35)
cv2.imshow('dst', dst)
cv2.waitKey(0)
