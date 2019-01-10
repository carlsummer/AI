import tensorflow as tf

batch_size = 20
# [None, 3072]
x = tf.placeholder(tf.float32, [batch_size, 3072])
y = tf.placeholder(tf.int64, [batch_size])
# [None],eg:[0,5,6,3]
x_image = tf.reshape(x, [-1, 3, 32, 32])
# 32 * 32
x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])

x_image_arr = tf.split(x_image,num_or_size_splits=batch_size,axis=0)
result_x_image_arr = []

for x_single_image in x_image_arr:
    # x_single_image: [1,32,32,3] -> [32,32,3]
    x_single_image = tf.reshape(x_single_image,[32,32,3])
    data_aug_1 = tf.image.random_flip_left_right(x_single_image)
    data_aug_2 = tf.image.random_brightness(data_aug_1,max_delta=0.3)
    # 在[lower, upper]的范围随机调整图的对比度。
    data_aug_3 = tf.image.random_contrast(data_aug_2,lower=0.2,upper=1.8)
    x_single_image = tf.reshape(data_aug_3,[1,32,32,3])
    result_x_image_arr.append(x_single_image)
result_x_images = tf.concat(result_x_image_arr,axis=0)

result_x_images = result_x_images /127.5 -1

# conv1: 神经元图， feature_map, 输出图像
conv1_1 = tf.layers.conv2d(result_x_images,
                           32,  # output channel number
                           (3, 3),  # kernel size
                           padding='same',
                           activation=tf.nn.relu,
                           name='conv1_1')
conv1_2 = tf.layers.conv2d(conv1_1,
                           32,  # output channel number
                           (3, 3),  # kernel size
                           padding='same',
                           activation=tf.nn.relu,
                           name='conv1_2')

# 16 * 16
pooling1 = tf.layers.max_pooling2d(conv1_2,
                                   (2, 2),  # kernel size
                                   (2, 2),  # stride
                                   name='pool1')

conv2_1 = tf.layers.conv2d(pooling1,
                           32,  # output channel number
                           (3, 3),  # kernel size
                           padding='same',
                           activation=tf.nn.relu,
                           name='conv2_1')
conv2_2 = tf.layers.conv2d(conv2_1,
                           32,  # output channel number
                           (3, 3),  # kernel size
                           padding='same',
                           activation=tf.nn.relu,
                           name='conv2_2')
# 8 * 8
pooling2 = tf.layers.max_pooling2d(conv2_2,
                                   (2, 2),  # kernel size
                                   (2, 2),  # stride
                                   name='pool2')

conv3_1 = tf.layers.conv2d(pooling2,
                           32,  # output channel number
                           (3, 3),  # kernel size
                           padding='same',
                           activation=tf.nn.relu,
                           name='conv3_1')
conv3_2 = tf.layers.conv2d(conv3_1,
                           32,  # output channel number
                           (3, 3),  # kernel size
                           padding='same',
                           activation=tf.nn.relu,
                           name='conv3_2')
# 4 * 4 * 32
pooling3 = tf.layers.max_pooling2d(conv3_2,
                                   (2, 2),  # kernel size
                                   (2, 2),  # stride
                                   name='pool3')
# [None, 4 * 4 * 32]
flatten = tf.layers.flatten(pooling3)

# [None, 4 * 4 * 32]
y_ = tf.layers.dense(flatten,10)

"""交叉熵损失函数"""
loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

# bool
predict = tf.argmax(y_, 1)  # 取第二维上取最大值
# [1,0,1,1,1,0,0,0]
correct_prediction = tf.equal(predict, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
