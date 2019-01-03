import tensorflow as tf

def inception_block(x,
                    output_channel_for_each_path,
                    name):
    """inception block implementation"""
    with tf.variable_scope(name):
        conv1_1 = tf.layers.conv2d(x,
                                   output_channel_for_each_path[0],
                                   (1,1),
                                   strides = (1,1),
                                   padding= "same",
                                   activation=tf.nn.relu,
                                   name="conv1_1")
        conv3_3 = tf.layers.conv2d(x,
                                   output_channel_for_each_path[1],
                                   (3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   activation=tf.nn.relu,
                                   name="conv3_3")
        conv5_5 = tf.layers.conv2d(x,
                                   output_channel_for_each_path[2],
                                   (5, 5),
                                   strides=(1, 1),
                                   padding="same",
                                   activation=tf.nn.relu,
                                   name="conv5_5")
        max_pooling = tf.layers.max_pooling2d(x,
                                              (2,2),
                                              (2,2),
                                              name="max_pooling")


# [None, 3072]
x = tf.placeholder(tf.float32, [None, 3072])
y = tf.placeholder(tf.int64, [None])
# [None],eg:[0,5,6,3]
x_image = tf.reshape(x, [-1, 3, 32, 32])
# 32 * 32
x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])


y_  = tf.layers.dense(flatten,10)

"""交叉熵损失函数"""
loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

# bool
predict = tf.argmax(y_, 1)  # 取第二维上取最大值
# [1,0,1,1,1,0,0,0]
correct_prediction = tf.equal(predict, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
