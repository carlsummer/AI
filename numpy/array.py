import numpy as np

#创建数组
a = np.array([1,2,3,4])
print(a)
#a的数据类型 <class 'numpy.ndarray'>
print(type(a))
#a.shape 数组a的个数 (4,)
print(a.shape)
#a.reshape 第一个参数定义为1行，-1是占位符表示3个数据
a = a.reshape((1,-1))
#(1, 4)
print(a.shape)
a = np.array([1,2,3,4,5,6])
a = a.reshape((2,-1))
print(a)
a = a.reshape((-1,2))
print(a)
print(a[2,0])
#zeros创建元素全部为0的数据 ,
a = np.zeros((3,3)) #创建一个3*3的数据全部为0的数组
print(a)
#ones 创建元素全部为1的数据 ,
a = np.ones((3,3))#创建一个3*3的数据全部为1的数组
print(a)
#full 创建指定维数的数组，和指定内容的 数组
a = np.full((2,3),1) #创建一个2*3的数据全部为1的数组
print(a)
#eye 创建从左上交到右下角都是1的数组,其余元素都是0
a = np.eye(3) #创建一个3*3从左上交到右下角都是1的数组,其余元素都是0
print(a)
#random.random 创建一个数值是随机的数组
a = np.random.random((3,4)) #创建一个3*4的里面的数值是随机的0到1的值
print(a)

a=np.array([[1,2,3,4],
           [5,6,7,8],
           [9,10,11,12]])
b = a[-2:,1:3]#从倒数第2行开始取，第1列到第3列之间的值包含第3列 [[ 6  7 ][10 11 ]]
print(b)
print(b.shape)

#arange产生指定范围的数组
print(np.arange(3,7))  #[3 4 5 6]

a[np.arange(3),1] += 10
print(a) #[[ 1 12  3  4][ 5 16  7  8][ 9 20 11 12]]

#获取数组中所有大于10的元素，返回boolean类型的数组
result_index  = a > 10
print(result_index) #[[False  True False False][False  True False False] [False  True  True  True]]
print(a[result_index]) #将a中所有大于10的数值取出来[12 16 20 11 12]

