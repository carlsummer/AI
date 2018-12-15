# svm本质 寻求一个最优的超平面 分类
# svm 核: line
# 身高体重 训练 预测
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1 准备data
rand1 = np.array([[155, 48], [159, 50], [164, 53], [168, 56], [172, 60]])
rand2 = np.array([[152, 53], [156, 55], [160, 56], [172, 64], [176, 65]])

# 2 label
label = np.array([[0], [0], [0], [0], [0], [1], [1], [1], [1], [1]])

# 3 data
data = np.vstack((rand1, rand2))
data = np.array(data, dtype='float32')

# svm 所有的数据都要有label
# [155,48] -- 0 女生 [152,53] -- 1 男生
# 监督学习 0 负样本 1 正样本

# 4训练
svm = cv2.ml.SVM_create()  # ml  机器学习模块 SVM_create() 创建
# 属性设置
svm.setType(cv2.ml.SVM_C_SVC)  # svm type
svm.setKernel(cv2.ml.SVM_LINEAR)  # line
svm.setC(0.01)
# 训练
result = svm.train(data, cv2.ml.ROW_SAMPLE, label)
# 预测
pt_data = np.vstack([[167, 55], [162, 57]])  # 0 女生 1 男生
pt_data = np.array(pt_data, dtype='float32')
print(pt_data)
(par1, par2) = svm.predict(pt_data)
print(par1, par2)
