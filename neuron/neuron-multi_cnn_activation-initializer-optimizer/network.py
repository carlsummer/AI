import tensorflow as tf
import os

# [None, 3072]
x = tf.placeholder(tf.float32, [None, 3072])
y = tf.placeholder(tf.int64, [None])
# [None],eg:[0,5,6,3]
x_image = tf.reshape(x, [-1, 3, 32, 32])
# 32 * 32
x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])


def convnet(inputs, activation, kernel_initializer):
    # conv1: 神经元图， feature_map, 输出图像
    conv1_1 = tf.layers.conv2d(x_image,
                               32,  # output channel number
                               (3, 3),  # kernel size
                               padding='same',
                               activation=activation,
                               kernel_initializer=kernel_initializer,
                               name='conv1_1')
    conv1_2 = tf.layers.conv2d(conv1_1,
                               32,  # output channel number
                               (3, 3),  # kernel size
                               padding='same',
                               activation=activation,
                               kernel_initializer=kernel_initializer,
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
                               activation=activation,
                               kernel_initializer=kernel_initializer,
                               name='conv2_1')
    conv2_2 = tf.layers.conv2d(conv2_1,
                               32,  # output channel number
                               (3, 3),  # kernel size
                               padding='same',
                               activation=activation,
                               kernel_initializer=kernel_initializer,
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
                               activation=activation,
                               kernel_initializer=kernel_initializer,
                               name='conv3_1')
    conv3_2 = tf.layers.conv2d(conv3_1,
                               32,  # output channel number
                               (3, 3),  # kernel size
                               padding='same',
                               activation=activation,
                               kernel_initializer=kernel_initializer,
                               name='conv3_2')
    # 4 * 4 * 32
    pooling3 = tf.layers.max_pooling2d(conv3_2,
                                       (2, 2),  # kernel size
                                       (2, 2),  # stride
                                       name='pool3')
    # [None, 4 * 4 * 32]
    flatten = tf.layers.flatten(pooling3)
    return flatten

#sigmoid:53.39% vs relu: 73.35% on 10k train
# tf.glorot_uniform_initializer: 76.53% 100k train.
# tf.truncated_normal_initializer: 78.04% 100k train
# tf.keras.initializers.he_normal: 71.52% 100k train
flatten = convnet(x_image, tf.nn.relu, tf.truncated_normal_initializer)
y_ = tf.layers.dense(flatten, 10)

"""交叉熵损失函数"""
loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

# bool
predict = tf.argmax(y_, 1)  # 取第二维上取最大值
# [1,0,1,1,1,0,0,0]
correct_prediction = tf.equal(predict, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

with tf.name_scope('train_op'):
    #train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
    # gradient descent: 12.35% train 100k
    #train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)
    #momentum: 35.75% train 100k
    #reason: 1. initializer incorrect, 2. 不充分的训练
    train_op = tf.train.MomentumOptimizer(learning_rate=1e-4,momentum=0.9).minimize(loss)

# with tf.name_scope("summary"):
#     variable_summary(conv1_1,'conv1_1')
#     variable_summary(conv1_2, 'conv1_2')
#     variable_summary(conv2_1, 'conv2_1')
#     variable_summary(conv2_2, 'conv2_2')
#     variable_summary(conv3_1, 'conv3_1')
#     variable_summary(conv3_2, 'conv3_2')

loss_summary = tf.summary.scalar("loss", loss)
accuracy_summary = tf.summary.scalar("accuracy", accuracy)

source_image = (x_image + 1) * 127.5
inputs_summary = tf.summary.image("inputs_image", source_image)

merged_summary = tf.summary.merge_all()
merged_summary_test = tf.summary.merge([loss_summary, accuracy_summary])

LOG_DIR = "."
