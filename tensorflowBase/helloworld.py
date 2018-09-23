# _*_ coding: UTF-8 _*_

import tensorflow as tf
import numpy as  np

hw = tf.constant("Hello World ! I love tensorflow")
sess = tf.Session()
print(sess.run(hw))
sess.close()

vector = np.array([1, 2, 3])
print(vector.shape)  # 形状是(3,)
print(vector.size)  # size是3
print(vector.ndim)  # 维度是1维

zeros = np.zeros((3, 4))  # 创建一个3行4列全零的数组
print(zeros)
ones = np.ones((5, 6))  # 创建一个5行6列的全1的数组
print(ones)
ident = np.eye(4)  # 创建一个对角线为1的数组
print(ident)

const = tf.constant(4)  # 创建一个常量
print(const)
var = tf.Variable(3)  # 创建一个变量
print(var)
# feed_dict={x:rand_array} 用来给placeholder定义的变量赋值，x = rand_array

c = tf.constant([[2, 3], [4, 5]], name="const1", dtype=tf.int64)
print(c)
sess = tf.Session()
print(sess.run(c))
if c.graph is tf.get_default_graph():
    print("The graph of c is default graph")
