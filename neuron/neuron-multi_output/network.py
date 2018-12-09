import tensorflow as tf

# [None, 3072]
x = tf.placeholder(tf.float32, [None, 3072])
# [None],eg:[0,5,6,3]
y = tf.placeholder(tf.int64, [None])

# （3072，10）
w = tf.get_variable('w', [x.get_shape()[-1], 10],
                    initializer=tf.random_normal_initializer(0, 1))
# (10,)
b = tf.get_variable("b", [10], initializer=tf.constant_initializer(0.0))

# [None,3072]*[3072,10] = [None,10]
y_ = tf.matmul(x, w) + b

# mean square loss
"""平方差损失函数
# e^x / sum(e^x)
# [[0.01,0.09,....,0.03],[]]
p_y = tf.nn.softmax(y_)
# 5 ->[0,0,0,0,1,0,0,0,0,0]
y_one_hot = tf.one_hot(y, 10, dtype=tf.float32)
loss = tf.reduce_mean(tf.square(y_one_hot - p_y))
"""

"""交叉熵损失函数"""
loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
# y_ ->softmax
# y->one_hot
# loss = ylogy_


"""
# [None,1]
p_y_1 = tf.nn.sigmoid(y_)  # 将y_生成值
# [None ,1]
y_reshaped = tf.reshape(y, (-1, 1))
y_reshaped_float = tf.cast(y_reshaped, tf.float32)

loss = tf.reduce_mean(tf.square(y_reshaped_float - p_y_1))
"""

# bool
predict = tf.argmax(y_, 1)  # 取第二维上取最大值
# [1,0,1,1,1,0,0,0]
correct_prediction = tf.equal(predict, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
