import numpy as np

a = np.array([1,2])
print(a.dtype) #int32
a = np.array([1.1,2.2])
print(a.dtype) #float64
a = np.array([1,2.2])
print(a.dtype) #float64
a = np.array([1.1,2.2],dtype=np.int64)
print(a) #[1 2]

#数学运算与常用函数
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[6,5]])
#加法
c = a+b
print(c) #[[6 8][9 9]]
print(np.add(a,b))#[[6 8][9 9]]
#减法
print(a-b) #[[-4 -4][-3 -1]]
print(np.subtract(a,b))#[[-4 -4][-3 -1]]
#乘法
print(a*b) #[[ 5 12][18 20]]
print(np.multiply(a,b))#[[ 5 12][18 20]]
#除法
print(a/b) #[[0.2        0.33333333][0.5        0.8       ]]
print(np.divide(a,b))#[[0.2        0.33333333][0.5        0.8       ]]
#开方
print(np.sqrt(a)) #[[1.         1.41421356][1.73205081 2.        ]]

b = np.array([[1,2,3],[4,5,6]])
print(a.dot(b)) # a[[1,2],[3,4]]    [[ 9 12 15][19 26 33]]

print(np.sum(a)) #求数组中所有元素的和10  1+2+3+4
print(np.sum(a,axis=0)) #求数组中每列所有元素的和[4 6]
print(np.sum(a,axis=1)) #求数组中每行所有元素的和[3 7]

print(np.mean(a)) #求数组中所有元素的平均值2.5
print(np.mean(a,axis=0)) #求数组中每列所有元素的平均值[2. 3.]
print(np.mean(a,axis=1)) #求数组中每行所有元素的平均值[1.5 3.5]

print(np.random.uniform(1,100)) #产生1到100的随机数

print(np.tile(a,(1,2))) #将数组a 显示为1行2列的大数组
#[[1 2 1 2]
# [3 4 3 4]]

print(a.argsort()) #对每一行进行从小到大排列

print(a.T) #转置
print(np.transpose(a)) #转置