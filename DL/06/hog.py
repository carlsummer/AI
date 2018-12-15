# 1 什么是hog》？特征 某个像素 某种运算
# 2 2·1 模块划分 2·2 梯度 方向 模版 2·3 bin 投影 2·4 每个模块hog
# 2·1 模块划分
# image（ppt） win（蓝色） block（红色） cell （绿色）（size）
# image》win〉block》cell
# block setp  win step cell bin
# win 特征计算最顶层单元 -》obj
# 1 win size 50*100 20*50    64*128
# 2 2.1 block 《win 2.2 win size w h / block （wh） 16*16
# 3 block step  如何win下滑动 8*8
# 4 计算block cout = （（64-16）/8+1）*（（128-16）/8+1）= 105 block
# 5 cell size 8*8
# 6 block = ？cell 16*16 = 2*2 = 》4cell  cell1-cell4
# 7 bin？

# 7 cell bin 梯度：运算
# 每个像素-》梯度 ：大小 f 方向 angle
# 0-360 /40 = 9块 = 9bin
# 1bin = 40 cell-》360-〉9bin
# hog特征维度：
# haar 值 hog 向量 （维度）-》完全描述 一个obj info all
# 维度 = 105*4*9=3780

# 2·2 梯度 方向 模版
# 像素都有一个梯度 》hog== win
# 特征模版-》haar类似
# 【1 0 -1】【【1】【0】【-1】】
# a = p1*1+p2*0+p3*(-1) = 相邻像素之差
# b = 上下像素之差
# f = 根号下（a方+b方）
# angle = arctan（a/b）

# bin 投影 梯度
# bin 0-360 9bin 0-40
# bin1 0-20 180-200
# ij f a = 10
# 0-20 center bin1 a=190 180 200 bin1
# f
# 25 bin1 bin2
# f1 = f*f（夹角） f2 = f*（1-f（夹角））  f（夹角）  0-1.0
# +1 hog

# 整体hog cell复用
# 3780
# 3780 《-win（block cell bin）
# 1《-bin
# cell0 cell3 bin0-bin8
# cell0: bin0 bin1 。。。bin8
# cell1: bin0 bin1 。。。bin8
# cell2: bin0 bin1 。。。bin8
# cell3: bin0 bin1 。。。bin8
# ij cell0 bin0=《f0，
# i+1 j cell0 bin0 = f1
# ij。。。。
# sumbin0（f0+f1.。）= bin0
# 权重累加
# ij bin0 bin1

# cell复用

# block 4个cell
# 【0】【】【】【3】
# cell0 bin0-bin9
# cellx0 cellx2 cellx4
# cellx0:ij-》bin bin+1
# cellx2：ij -》 cell2 cell3 -》bin bin+1 bin bin+1
# cellx4：ij

# 【cell 9】【4cell】【105】 = 3780

# 【3780】hog svm line训练【3780】
# 。hog*svm = 值
# 值》T 目标obj

# 1 样本 2 训练 3 test 预测
# 1 样本
# 1.1 pos 正样本 包含所检测目标 neg 不包含obj
# 1.2 如何获取样本 1 网络 2 公司内部 3 自己收集
# 一个好的样本 远胜过一个 复杂的神经网络 （K w）（M）
# 1.1 网络公司 样本：1张图 1元  贵
# 1.2 网络 爬虫 自己爬
# 1.3 公司： 很多年积累（mobileeye ADAS 99%） 红外图像
# 1.4 自己收集 视频 100 30 = 3000
# 正样本：尽可能的多样  环境 干扰
# 820 pos neg 1931 1:2 1:3
# name

# 训练
# 1 参数 2hog 3 svm 4 computer hog 5 label 6 train 7 pred 8 draw
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1 par
PosNum = 820
NegNum = 1931
winSize = (64, 128)
blockSize = (16, 16)  # 105
blockStride = (8, 8)  # 4 cell
cellSize = (8, 8)
nBin = 9  # 9 bin  3780

# 2 hog create hog 1 win 2 block 3 blockStride 4 cell 5 bin
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nBin)
# 3 svm
svm = cv2.ml.SVM_create()
# 4 computer hog
featureNum = int(((128 - 16) / 8 + 1) * ((64 - 16) / 8 + 1) * 4 * 9)  # 3780
featureArray = np.zeros(((PosNum + NegNum), featureNum), np.float32)
labelArray = np.zeros(((PosNum + NegNum), 1), np.int32)
# svm 监督学习 样本 标签 svm -》image hog
for i in range(0, PosNum):
    fileName = './pos/' + str(i + 1) + '.jpg'
    img = cv2.imread(fileName)
    hist = hog.compute(img, (8, 8))  # 3780
    for j in range(0, featureNum):
        featureArray[i, j] = hist[j]
    # featureArray hog [1,:] hog1 [2,:]hog2
    labelArray[i, 0] = 1
    # 正样本 label 1

for i in range(0, NegNum):
    fileName = './neg/' + str(i + 1) + '.jpg'
    img = cv2.imread(fileName)
    hist = hog.compute(img, (8, 8))  # 3780
    for j in range(0, featureNum):
        featureArray[i + PosNum, j] = hist[j]
    # featureArray hog [1,:] hog1 [2,:]hog2
    labelArray[i + PosNum, 0] = -1
    ## 负样本 label -1

svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(0.01)
# 6 train
ret = svm.train(featureArray, cv2.ml.ROW_SAMPLE, labelArray)
# 7 myHog ：《-myDetect
# myDetect-《resultArray  rho
# myHog-》detectMultiScale

# 7 检测  核心：create Hog -》 myDetect—》array-》
# resultArray-》resultArray = -1*alphaArray*supportVArray
# rho-》svm-〉svm.train
alpha = np.zeros((1), np.float32)
rho = svm.getDecisionFunction(0, alpha)
print(rho)
print(alpha)
alphaArray = np.zeros((1, 1), np.float32)
supportVArray = np.zeros((1, featureNum), np.float32)
resultArray = np.zeros((1, featureNum), np.float32)
alphaArray[0, 0] = alpha
resultArray = -1 * alphaArray * supportVArray
# detect
myDetect = np.zeros((3781), np.float32)
for i in range(0, 3780):
    myDetect[i] = resultArray[0, i]
myDetect[3780] = rho[0]
# 构建hog # rho svm （判决）
myHog = cv2.HOGDescriptor()
myHog.setSVMDetector(myDetect)
# load
imageSrc = cv2.imread('Test2.jpg', 1)
# (8,8) win
objs = myHog.detectMultiScale(imageSrc, 0, (8, 8), (32, 32), 1.05, 2)
# xy wh 三维 最后一维
x = int(objs[0][0][0])
y = int(objs[0][0][1])
w = int(objs[0][0][2])
h = int(objs[0][0][3])
# 绘制展示
cv2.rectangle(imageSrc, (x, y), (x + w, y + h), (255, 0, 0), 2)
cv2.imshow("dst", imageSrc)
cv2.waitKey(0)
