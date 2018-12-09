import cv2
import numpy as np
import random

img = cv2.imread('2.jpg', 1)
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
dst = np.zeros((height, width, 3), np.uint8)
mm = 8
for m in range(0, height - mm):
    for n in range(0, width - mm):
        index = int(random.random() * 8)  # 0+8
        (b, g, r) = img[m + index, n + index]
        dst[m, n] = (b, g, r)
cv2.imshow('dst', dst)
cv2.imwrite('2v.jpg', dst)
cv2.waitKey(0)
