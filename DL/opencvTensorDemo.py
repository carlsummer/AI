# opencv tensorflow
# 类比 语法 api 原理
# 基础数据类型 运算符 流程 字典 数组
import tensorflow as tf

data1 = tf.constant(2)
data2 = tf.Variable(10, name="var")
"""
print(data1)
print(data2)
sess = tf.Session()
print(sess.run(data1))
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(data2))
sess.close()
"""
data3 = tf.constant(6)
dataAdd = tf.add(data1, data3)
dataMul = tf.multiply(data1, data3)
dataSub = tf.subtract(data1, data3)
dataDiv = tf.divide(data1, data3)
dataAdd2 = tf.add(data1, data2)
dataCopy = tf.assign(data2, dataAdd2)  # dataAdd -> data2
dataMul2 = tf.multiply(data1, data2)
dataSub2 = tf.subtract(data1, data2)
dataDiv2 = tf.divide(data1, data2)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(dataAdd))
    print(sess.run(dataMul))
    print(sess.run(dataSub))
    print(sess.run(dataDiv))
    print(sess.run(dataAdd2))
    print(sess.run(dataMul2))
    print(sess.run(dataSub2))
    print(sess.run(dataDiv2))
    print("sess.run(dataCopy)", sess.run(dataCopy)) #8->data2
    print(sess.run(data2))
    print("dataCopy.eval()", dataCopy.eval())#8+6->14->data2=14
    print("tf.get_default_session()", tf.get_default_session().run(dataCopy))
