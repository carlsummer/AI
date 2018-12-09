# 100->200 x
# 100->300 y
import cv2
import numpy as np

img = cv2.imread("../image0.jpg", 1)
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
mode = imgInfo[2]
dst1 = img[100:200, 100:300]
cv2.imshow("dst1", dst1)
# 1 opencv API resize 2 算法原理 3 源码

####
matShift = np.float32([[1, 0, 100], [0, 1, 200]])  # 水平方向移动了100 垂直方向移动了200
dst = cv2.warpAffine(img, matShift, (height, width))
# 移位 矩阵
cv2.imshow("dst", dst)

# [1,0,100],[0,1,200] 2*2 2*1
# [[1,0],[0,1]]  2*2  A
# [[100],[200]] 2*1   B
# xy C
# A*C+B = [[1*x+0*y],[0*x+1*y]]+[[100],[200]]
# = [[x+100],[y+200]]

# (10,20)->(110,120)
newImgInfo = (height * 2, width, mode)
dst3 = np.zeros(newImgInfo, np.uint8)
for i in range(0, height):
    for j in range(0, width):
        dst3[i, j] = img[i, j]
        # x y = 2 * h -y -1
        dst3[height * 2 - i - 1, j] = img[i, j]
for i in range(0, width):
    dst3[height, i] = (0, 0, 255)  # RGB
cv2.imshow('dst3', dst3)

matSrc = np.float32([[0, 0], [0, height - 1], [width - 1, 0]])
matDst = np.float32([[50, 50], [300, height - 200], [width - 300, 100]])
# 组合
matAffline = cv2.getAffineTransform(matSrc, matDst)
dst4 = cv2.warpAffine(img, matAffline, (width, height))
cv2.imshow("dst4", dst4)
# 2 *3
matRotate = cv2.getRotationMatrix2D((height * 0.5, width * 0.5), 45, 0.5)  # 1中心点，2旋转的角度，3缩放的比例
dst5 = cv2.warpAffine(img, matRotate, (height, width))
cv2.imshow("dst5", dst5)
cv2.waitKey(0)
