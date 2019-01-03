import tensorflow as tf


def residual_block(x, output_channel):
    """residual connection implementation"""
    input_channel = x.get_shape().as_list()[-1]
    if input_channel * 2 == output_channel:
        increase_dim = True
        strides = (2, 2)
    elif input_channel == output_channel:
        increase_dim = False
        strides = (1, 1)
    else:
        raise Exception("input channel can't match output channel")
    conv1 = tf.layers.conv2d(x,
                             output_channel,
                             (3, 3),
                             strides=strides,
                             padding="same",
                             activation=tf.nn.relu,
                             name="conv1")
    conv2 = tf.layers.conv2d(conv1,
                             output_channel,
                             (3, 3),
                             strides=(1, 1),
                             padding="same",
                             activation=tf.nn.relu,
                             name="conv2")
    if increase_dim:
        # 采样形状和步长一样可以使得样本减少一半
        pooled_x = tf.layers.average_pooling2d(x,
                                               (2, 2),
                                               (2, 2),
                                               padding="valid")
        padded_x = tf.pad(pooled_x,
                          [[0, 0],
                           [0, 0],
                           [0, 0],
                           [input_channel // 2, input_channel // 2]])
    else:
        padded_x = x
    output_x = conv2 + padded_x
    return output_x


# x 输入快，num_residual_blocks残差连接块，num_subsampling需要做多少次亚采样 num_filter_base通道数的一个base
# class_num为了泛化处理，添加这个可以接收多种不同的数据集
def res_net(x,
            num_residual_blocks,
            num_filter_base,
            class_num):
    """residual network implementation"""
    num_subsampling = len(num_residual_blocks)
    layers = []
    # x: [None,width,height,channel] -> [width,height,channel]
    input_size = x.get_shape().as_list()[1:]
    with tf.variable_scope("conv0"):
        conv0 = tf.layers.conv2d(x, num_filter_base, (3, 3),padding="same", strides=(1, 1), activation=tf.nn.relu, name="conv0")
        layers.append(conv0)
    for sample_id in range(num_subsampling):
        for i in range(num_residual_blocks[sample_id]):
            with tf.variable_scope("conv%d_%d" % (sample_id,i)):
                conv = residual_block(
                    layers[-1],
                    num_filter_base * (2** sample_id))
                layers.append(conv)
    multiplier = 2 ** (num_subsampling - 1)
    assert layers[-1].get_shape().as_list()[1:] \
       == [input_size[0] / multiplier,
           input_size[1] / multiplier,
           num_filter_base * multiplier]
    with tf.variable_scope('fc'):
        # layer[-1].shape : [None,width,height,channel]
        # kernal_size: image_width,image_height
        global_pool = tf.reduce_mean(layers[-1],[1,2])
        logits = tf.layers.dense(global_pool,class_num)
        layers.append(logits)
    return layers[-1]


# [None, 3072]
x = tf.placeholder(tf.float32, [None, 3072])
y = tf.placeholder(tf.int64, [None])
# [None],eg:[0,5,6,3]
x_image = tf.reshape(x, [-1, 3, 32, 32])
# 32 * 32
x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])

# [None, 4 * 4 * 32]
y_ = res_net(x_image,[2,3,2],32,10)

"""交叉熵损失函数"""
loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

# bool
predict = tf.argmax(y_, 1)  # 取第二维上取最大值
# [1,0,1,1,1,0,0,0]
correct_prediction = tf.equal(predict, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
