import cv2
img = cv2.imread('./image1.jpg')
imgInfo = img.shape
size = (imgInfo[1],imgInfo[0])
print(size)
# windows下使用DIVX
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
videoWrite = cv2.VideoWriter('2.avi',-1,5,size,True)# 写入对象 1 file name
# 2 编码器 3 帧率 4 size
for i in range(1,11):
    fileName = './image'+str(i)+'.jpg'
    img = cv2.imread(fileName)
    videoWrite.write(img)# 写入方法 1 jpg data
print('end!')