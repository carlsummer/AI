# placeholder
import tensorflow as tf
import numpy as np

"""
data1 = tf.placeholder(tf.float32)
data2 = tf.placeholder(tf.float32)
dataAdd = tf.add(data1, data2)
with tf.Session() as sess:
    print(sess.run(dataAdd, feed_dict={data1: 2, data2: 3}))
"""

"""
# 类比 数组M行N列 [] 内部[] [里面 列数据] [] 中括号整体 行数
# [[6,6],[6,6]]
data1 = tf.constant([[6, 6]])
data2 = tf.constant([[2],
                     [2]])
data3 = tf.constant([[3, 3]])
data4 = tf.constant([[1, 2],
                     [3, 4],
                     [5, 6]])
print(data4.shape)  # 打印矩阵的维度
matmul = tf.matmul(data1, data2)
matadd = tf.add(data1, data3)
with tf.Session() as sess:
    print(sess.run(data4))
    print(sess.run(data4[0]))  # 打印某一行
    print(sess.run(data4[:, 0]))  # 打印某一列
    print(sess.run(matmul))
    print(sess.run(matadd))
    print(sess.run([matmul, matadd]))
"""

"""
mat0 = tf.constant([[0, 0, 0], [0, 0, 0]])
mat1 = tf.zeros([2, 3])
mat2 = tf.ones([3, 2])
mat3 = tf.fill([2, 3], 15)
with tf.Session() as sess:
    print(sess.run([mat0, mat1, mat2,mat3]))
"""

"""
mat1 = tf.constant([[2], [3], [4]])
mat2 = tf.zeros_like(mat1)
mat3 = tf.linspace(0.0, 2.0, 11)
mat4 = tf.random_uniform([2, 3], -1, 2)
with tf.Session() as sess:
    print(sess.run([mat1, mat2, mat3, mat4]))
"""

data1 = np.array([1, 2, 3, 4])
print(data1)
data2 = np.array([[1, 2], [3, 4]])
print(data2)
# 维度
print(data1.shape, data2.shape)
# zero ones
print(np.zeros([2, 3]), np.ones([2, 2]))
# 改查
data2[1, 0] = 5
print(data2)
print(data2[1, 1])
# 基本运算
data3 = np.ones([2, 3])
print(data3 * 2)  # 对应相乘
print(data3 / 3)
print(data3 + 2)
# 矩阵+*
data4 = np.array([[1, 2, 3], [4, 5, 6]])
print(data3 + data4)
print(data3 * data4)
