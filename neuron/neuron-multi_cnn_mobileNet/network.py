import tensorflow as tf


def separable_conv_block(x,
                         output_channel_number,
                         name):
    """inception block implementation"""
    with tf.variable_scope(name):
        input_channel = x.get_shape().as_list()[-1]
        # channel_wise_x: [channel1,channel2,....]
        channel_wise_x = tf.split(x, input_channel, axis=3)
        output_channels = []
        for i in range(len(channel_wise_x)):
            output_channel = tf.layers.conv2d(channel_wise_x[i],
                                              1,
                                              (3, 3),
                                              strides=(1, 1),
                                              padding="same",
                                              activation=tf.nn.relu,
                                              name="conv_%d" % i)
            output_channels.append(output_channel)
        concat_layer = tf.concat(output_channels, axis=3)
        conv1_1 = tf.layers.conv2d(concat_layer,
                                   output_channel_number,
                                   (1, 1),
                                   strides=(1, 1),
                                   padding="same",
                                   activation=tf.nn.relu,
                                   name="conv1_1")
        return conv1_1


# [None, 3072]
x = tf.placeholder(tf.float32, [None, 3072])
y = tf.placeholder(tf.int64, [None])
# [None],eg:[0,5,6,3]
x_image = tf.reshape(x, [-1, 3, 32, 32])
# 32 * 32
x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])

# conv1: 神经元图，feature_map，输出图像
conv1 = tf.layers.conv2d(x_image,
                         32,  # output channel number
                         (3, 3),  # kernel size
                         padding='same',
                         activation=tf.nn.relu,
                         name='conv1')
# 16*16
pooling1 = tf.layers.max_pooling2d(conv1,
                                   (2, 2),  # kernel size
                                   (2, 2),  # stride
                                   name="pool1")
separable_2a = separable_conv_block(pooling1,
                                    32,
                                    name="separable_2a")
separable_2b = separable_conv_block(separable_2a,
                                    32,
                                    name="separable_2b")
pooling2 = tf.layers.max_pooling2d(separable_2b,
                                   (2, 2),  # kernel size
                                   (2, 2),  # stride
                                   name="pool2")
separable_3a = separable_conv_block(pooling2,
                                    32,
                                    name="separable_3a")
separable_3b = separable_conv_block(separable_3a,
                                    32,
                                    name="separable_3b")
pooling3 = tf.layers.max_pooling2d(separable_3b,
                                   (2, 2),  # kernel size
                                   (2, 2),  # stride
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
