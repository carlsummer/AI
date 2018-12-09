import cv2
import numpy as np


def ImageHist(image, type):
    color = (255, 255, 255)
    windowName = 'Gray'
    if type == 31:
        color = (255, 0, 0)
        windowName = 'B Hist'
    elif type == 32:
        color = (0, 255, 0)
        windowName = "G Hist"
    elif type == 33:
        color = (0, 0, 255)
        windowName = "R Hist"
    # 1 图片 2 通道 3 蒙版 4 表示有多少个柱状 5 表明直方图中各个像素的值
    hist = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])
    minV, maxV, minL, maxL = cv2.minMaxLoc(hist)
    histImg = np.zeros([256, 256, 3], np.uint8)
    for h in range(256):
        intenNormal = int(hist[h] * 256 / maxV)  # 归一化
        cv2.line(histImg, (h, 256), (h, 256 - intenNormal), color)
    cv2.imshow(windowName, histImg)
    return histImg


img = cv2.imread("../image0.jpg", 1)
channels = cv2.split(img)  # RGB -R G B
for i in range(0, 3):
    ImageHist(channels[i], 31 + i)
cv2.waitKey(0)
