import os

from mnist import input_data
from mnist import model
import tensorflow as tf

#one_hot
data = input_data.read_data_sets('MNIST_data', one_hot=True)

# with 随着 create model
with tf.variable_scope("regression"):
    # tf.placeholder(dtype, shape=None, name=None)
    # 此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值
    # dtype：数据类型。常用的是tf.float32,tf.float64等数值类型
    # shape：数据形状。默认是None，就是一维值，也可以是多维，比如[2,3], [None, 3]表示列是3，行不定
    # name：名称。
    x = tf.placeholder(tf.float32, [None, 784])
    y, variables = model.regression(x)

# train
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(variables)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(1000):
        batch_xs, batch_ys = data.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    print(sess.run(accuracy, feed_dict={x: data.test.images, y_: data.test.labels}))

    path = saver.save(
        sess, os.path.join(os.path.dirname(__file__), 'data', 'regression.ckpt'),
        write_meta_graph=False, write_state=False)
    print("Saved:", path)
