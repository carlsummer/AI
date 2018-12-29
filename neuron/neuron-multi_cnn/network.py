import tensorflow as tf

# [None, 3072]
x = tf.placeholder(tf.float32, [None, 3072])
y = tf.placeholder(tf.int64, [None])
# [None],eg:[0,5,6,3]
x_image = tf.reshape(x, [-1, 3, 32, 32])
# 32 * 32
x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])

#conv1: 神经元图，feature_map，输出图像
conv1 = tf.layers.conv2d(x_image,
                         32, #output channel number
                         (3,3), #kernel size
                         padding='same',
                         activation=tf.nn.relu,
                         name='conv1')
# 16*16
pooling1 = tf.layers.max_pooling2d(conv1,
                                   (2,2), # kernel size
                                   (2,2),# stride
                                    name="pool1")
conv2 = tf.layers.conv2d(pooling1,
                         32, #output channel number
                         (3,3), #kernel size
                         padding='same',
                         activation=tf.nn.relu,
                         name='conv2')
# 8*8
pooling2 = tf.layers.max_pooling2d(conv2,
                                   (2,2), # kernel size
                                   (2,2),# stride
                                    name="pool2")
conv3 = tf.layers.conv2d(pooling2,
                         32, #output channel number
                         (3,3), #kernel size
                         padding='same',
                         activation=tf.nn.relu,
                         name='conv3')
# 4*4
pooling3 = tf.layers.max_pooling2d(conv3,
                                   (2,2), # kernel size
                                   (2,2),# stride
                                    name="pool3")
# [None, 4 * 4 * 32]
flatten = tf.layers.flatten(pooling3)
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
