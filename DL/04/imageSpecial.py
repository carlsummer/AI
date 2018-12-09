# imread
# 方法1 imread
import cv2
import numpy as np

img0 = cv2.imread("../image0.jpg", 0)
img1 = cv2.imread("../image0.jpg", 1)
print(img0.shape)
print(img1.shape)
cv2.imshow('img0', img0)
# 方法2 cvtColor
img = cv2.imread("../image0.jpg", 1)
dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 颜色空间转换 1 data 2 转换的类型
cv2.imshow("cvt", dst)

# 0-255 255-当前
# img = cv2.imread("2.jpg", 1)
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst1 = np.zeros((height, width, 1), np.uint8)
for i in range(0, height):
    for j in range(0, width):
        grayPixel = gray[i, j]
        dst1[i, j] = 255 - grayPixel
cv2.imshow("dst1", dst1)

# RGB 255-R=newR
dst2 = np.zeros((height, width, 3), np.uint8)
for i in range(0, height):
    for j in range(0, width):
        (b, g, r) = img[i, j]
        dst2[i, j] = (255 - b, 255 - g, 255 - r)
cv2.imwrite("22.jpg", dst2)

for m in range(100, 300):
    for n in range(100, 200):
        # pixel -> 10*10
        if m % 10 == 0 and n % 10 == 0:
            for i in range(0, 10):
                for j in range(0, 10):
                    (b, g, r) = img[m, n]
                    img[i + m, j + n] = (b, g, r)
cv2.imshow("jj", img)
cv2.waitKey(0)
