import cv2
import numpy as np
import random
import math

"""
img = cv2.imread("../image0.jpg", 1)
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
"""
# newP = gray0-gray1+150  gray0-gray1相邻2个灰度等级差为了灰度突变和边缘，+150是为了突出灰度等级
"""
dst = np.zeros((height, width, 1), np.uint8)
for i in range(0, height):
    for j in range(0, width - 1):
        grayP0 = int(gray[i, j])
        grayP1 = int(gray[i, j + 1])
        newP = grayP0 - grayP1 + 150
        if newP > 255:
            newP = 25
        if newP < 0:
            newP = 0
        dst[i, j] = newP
cv2.imshow('dst', dst)
"""

# rgb -》RGB new “蓝色”
# b=b*1.5
# g = g*1.3
"""
dst = np.zeros((height, width, 3), np.uint8)
for i in range(0, height):
    for j in range(0, width):
        (b, g, r) = img[i, j]
        b = b * 1.5
        g = g * 1.3
        if b > 255:
            b = 255
        if g > 255:
            g = 255
        dst[i, j] = (b, g, r)
cv2.imshow("dst", dst)
"""

# 1 gray 2 7*7 10*10 3 0-255 256 4 640-63 64-127
# 3 10 0-63 99
# 4 count 5 dst = result
"""
dst = np.zeros((height, width, 3), np.uint8)
for i in range(4, height - 4):
    for j in range(4, width - 4):
        array1 = np.zeros(8, np.uint8)
        for m in range(-4, 4):
            for n in range(-4, 4):
                p1 = int(gray[i + m, j + n] / 32)
                array1[p1] = array1[p1] + 1
        currentMax = array1[0]
        l = 0
        for k in range(0, 8):
            if currentMax < array1[k]:
                currentMax = array1[k]
                l = k
        # 简化 均值
        for m in range(-4, 4):
            for n in range(-4, 4):
                if gray[i + m, j + n] >= (l * 32) and gray[i + m, j + n] <= ((l + 1) * 32):
                    (b, g, r) = img[i + m, j + n]
            dst[i, j] = (b, g, r)
    cv2.imshow('dst', dst)
"""

"""
newImageInfo = (500, 500, 3)
dst = np.zeros(newImageInfo, np.uint8)
# line
# 绘制线段 1 dst 2 begin 3 end 4 color
cv2.line(dst, (100, 100), (400, 400), (0, 0, 255))
# 5 line width
cv2.line(dst, (100, 200), (400, 200), (0, 255, 255), 20)
# 6 line type
cv2.line(dst, (100, 300), (400, 300), (0, 255, 0), 20, cv2.LINE_AA)

cv2.line(dst, (200, 150), (50, 250), (25, 100, 255))
cv2.line(dst, (50, 250), (400, 380), (25, 100, 255))
cv2.line(dst, (400, 380), (200, 150), (25, 100, 255))

cv2.imshow("dst", dst)
"""

"""
newImageInfo = (500, 500, 3)
dst = np.zeros(newImageInfo, np.uint8)
# 1 图片数据 2 左上角 3 右上角 4颜色 5填充fill -1 表示填充 >0 表示border的大小
cv2.rectangle(dst, (50, 100), (200, 300), (255, 0, 0), 5)
# 1 图片数据 2 中心点 3 r半径 4 color颜色 5 填充fill -1 表示填充 >0 表示border的大小
cv2.circle(dst, (250, 250), (50), (0, 255, 0), 2)
# 1图片数据 2 中心点 3 轴的大小 4 angle 5begin 6 end 7 color 8  填充fill -1 表示填充 >0 表示border的大小
cv2.ellipse(dst, (256, 256), (150, 100), 0, 0, 180, (255, 255, 0), -1)

points = np.array([[150, 50], [140, 140], [200, 170], [250, 250], [150, 50]], np.int32)
print(points.shape)
points = points.reshape((-1, 1, 2))
print(points.shape)
cv2.polylines(dst, [points], True, (0, 255, 255))
cv2.imshow("dst", dst)
"""

"""
img = cv2.imread("../image0.jpg", 1)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.rectangle(img, (200, 100), (500, 400), (0, 255, 0), 3)
# 1 图片 2 文字 3 写入的坐标 4 字体 5 字体大小 6 color 7 粗细 8 line type
cv2.putText(img, "this is flow", (100, 300), font, 1, (200, 100, 255), 2, cv2.LINE_AA)
cv2.imshow("dst", img)
"""

img = cv2.imread("../image0.jpg", 1)
height = int(img.shape[0] * 0.2)
width = int(img.shape[1] * 0.2)
imgResize = cv2.resize(img, (width, height))
for i in range(0, height):
    for j in range(0, width):
        img[i + 200, j + 350] = imgResize[i, j]
cv2.imshow('src', img)
cv2.waitKey(0)
