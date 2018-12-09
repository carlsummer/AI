# 1 load 2 info 3 resize 4 check
import cv2

img = cv2.imread("../image0.jpg", 1)
imgInfo = img.shape
print(imgInfo)
height = imgInfo[0]
width = imgInfo[1]
mode = imgInfo[2]
# I 放大 缩小 II 等比例 非等比例
dstHeight = int(height * 0.5)
dstWidth = int(width * 0.5)
# 最近临域插值 双线性插值 像素关系重采样 立方插值
dst = cv2.resize(img, (dstWidth, dstHeight))
cv2.imshow("image", dst)

# 最近临域插值 双线性插值 原理
# src 10*20 dst 5*10
# dst<-src
# (1,2) <- (2,4)
# dst x 1 -> src x 2 newX
# newX = x*(src 行/目标 行) newX = 1*（10/5） = 2
# newY = y*(src 列/目标 列) newY = 2*（20/10）= 4
# 12.3 = 12

# 双线性插值
# A1 = 20% 上+80%下 A2
# B1 = 30% 左+70%右 B2
# 1 最终点  = A1 30% + A2 70%
# 2 最终点  = B1 20% + B2 80%

cv2.waitKey(0)
