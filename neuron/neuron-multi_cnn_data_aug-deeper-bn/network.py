import tensorflow as tf

batch_size = 20
# [None, 3072]
x = tf.placeholder(tf.float32, [batch_size, 3072])
y = tf.placeholder(tf.int64, [batch_size])
is_training = tf.placeholder(tf.bool,[])
# [None],eg:[0,5,6,3]
x_image = tf.reshape(x, [-1, 3, 32, 32])
# 32 * 32
x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])

x_image_arr = tf.split(x_image, num_or_size_splits=batch_size, axis=0)
result_x_image_arr = []

for x_single_image in x_image_arr:
    # x_single_image: [1,32,32,3] -> [32,32,3]
    x_single_image = tf.reshape(x_single_image, [32, 32, 3])
    data_aug_1 = tf.image.random_flip_left_right(x_single_image)
    data_aug_2 = tf.image.random_brightness(data_aug_1, max_delta=0.3)
    # 在[lower, upper]的范围随机调整图的对比度。
    data_aug_3 = tf.image.random_contrast(data_aug_2, lower=0.2, upper=1.8)
    x_single_image = tf.reshape(data_aug_3, [1, 32, 32, 3])
    result_x_image_arr.append(x_single_image)
result_x_images = tf.concat(result_x_image_arr, axis=0)

normal_result_x_images = result_x_images / 127.5 - 1


def conv_wrapper(inputs, name, is_training,output_channel=32, kernel_size=(3, 3), activation=tf.nn.relu, padding="same"):
    """wrapper of tf.layers.conv2d"""
    # without bn : conv -> activation
    # with batch normalization: conv -> bn -> activation
    with tf.name_scope(name):
        conv2d = tf.layers.conv2d(inputs,
                                  output_channel,
                                  kernel_size,
                                  padding=padding,
                                  activation=None,
                                  name="conv1")
        bn = tf.layers.batch_normalization(conv2d,training=is_training)

        return activation(bn)


def pooling_wrapper(inputs, name):
    return tf.layers.max_pooling2d(inputs, (2, 2), (2, 2), name=name)


# conv1: 神经元图， feature_map, 输出图像
conv1_1 = conv_wrapper(normal_result_x_images, 'conv1_1',is_training)
conv1_2 = conv_wrapper(conv1_1, 'conv1_2',is_training)
conv1_3 = conv_wrapper(conv1_2, 'conv1_3',is_training)
# 16 * 16
pooling1 = pooling_wrapper(conv1_3, 'pool1')

conv2_1 = conv_wrapper(pooling1, 'conv2_1',is_training)
conv2_2 = conv_wrapper(conv2_1, 'conv2_2',is_training)
conv2_3 = conv_wrapper(conv2_2, 'conv2_3',is_training)
# 8 * 8
pooling2 = pooling_wrapper(conv2_3, 'pool2')

conv3_1 = conv_wrapper(pooling2, 'conv3_1',is_training)
conv3_2 = conv_wrapper(conv3_1, 'conv3_2',is_training)
conv3_3 = conv_wrapper(conv3_2, 'conv3_3',is_training)
# 4 * 4 * 32
pooling3 = pooling_wrapper(conv3_3, 'pool3')

# [None, 4 * 4 * 32]
flatten = tf.layers.flatten(pooling3)

# [None, 4 * 4 * 32]
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