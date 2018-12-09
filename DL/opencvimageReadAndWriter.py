import cv2

# 1文件读取 2封装格式解析 3数据解码 4数据加载
img = cv2.imread('image0.jpg', 1)
cv2.imwrite("image1.png", img)
cv2.imwrite("imageTest.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 50])  # 0将图片压缩为10倍，有损压缩
cv2.imshow("image", img)
# jpg png 1 文件头 2 文件数据

# 压缩为png 的好处可以无损，改变透明度属性
cv2.imwrite("imageTest2.jpg", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
# jpg 0 压缩比高0-100 png 0 压缩比低 0-9
# 像素，RGB,颜色深度8bit 0-255
# 图片宽高640*480表示横向有640个像素点纵向有480个像素点
# 1.14M = 720*547*3*8 bit / 8 （B）
# png 图片 RGB alpha
# RGB bgr->第一个排的是蓝色

# 获取具体像素点的b g r
(b, g, r) = img[100, 100]
print(b, g, r)

# 读取10，100 ==== 110 100
for i in range(1, 100):
    img[10 + i, 100] = (255, 0, 0)

cv2.imshow('image2', img)  # 显示图片
cv2.waitKey(0)  # delay：延迟多少毫秒继续执行
