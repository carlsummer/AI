import tensorflow as tf


def inception_block(x,
                    output_channel_for_each_path,
                    name):
    """inception block implementation"""
    with tf.variable_scope(name):
        conv1_1 = tf.layers.conv2d(x,
                                   output_channel_for_each_path[0],
                                   (1, 1),
                                   strides=(1, 1),
                                   padding="same",
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
                                              (2, 2),
                                              (2, 2),
                                              name="max_pooling")
    max_pooling_shape = max_pooling.get_shape().as_list()[1:]
    input_shape = x.get_shape().as_list()[1:]
    width_padding = (input_shape[0] - max_pooling_shape[0]) // 2
    height_padding = (input_shape[1] - max_pooling_shape[1]) // 2
    padded_pooling = tf.pad(max_pooling,
                            [[0, 0],
                             [width_padding, width_padding],
                             [height_padding, height_padding],
                             [0, 0]])
    concat_layer = tf.concat([conv1_1, conv3_3, conv5_5, padded_pooling], axis=3)
    return concat_layer


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

inception_2a =  inception_block(pooling1,
                                [16,16,16],
                                name="inception_2a")
inception_2b =  inception_block(inception_2a,
                                [16,16,16],
                                name="inception_2b")
pooling2 = tf.layers.max_pooling2d(inception_2b,
                                   (2,2), # kernel size
                                   (2,2),# stride
                                    name="pool2")
inception_3a =  inception_block(pooling2,
                                [16,16,16],
                                name="inception_3a")
inception_3b =  inception_block(inception_3a,
                                [16,16,16],
                                name="inception_3b")
pooling3 = tf.layers.max_pooling2d(inception_3b,
                                   (2,2), # kernel size
                                   (2,2),# stride
                                    name="pool3")
flatten = tf.layers.flatten(pooling3)
y_ = tf.layers.dense(flatten, 10)

"""交叉熵损失函数"""
loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

# bool
predict = tf.argmax(y_, 1)  # 取第二维上取最大值
# [1,0,1,1,1,0,0,0]
correct_prediction = tf.equal(predict, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
