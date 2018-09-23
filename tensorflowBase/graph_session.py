# _*_ coding: UTF-8 _*_

import tensorflow as tf

# 创建两个常量Tensor
const1 = tf.constant([[2, 2]])
const2 = tf.constant([[4], [4]])

multiple = tf.matmul(const1, const2)  # 矩阵相乘
print(multiple)
sess = tf.Session()
result = sess.run(multiple)
print(result)
sess.close()

# 第二种方法来创建和关闭Session
with tf.Session() as sess:
    result2 = sess.run(multiple)
    print("Multiple的结果是：", result2)
