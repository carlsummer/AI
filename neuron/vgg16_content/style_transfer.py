import os
import math
import numpy as np
import tensorflow as tf
from PIL import Image
import time

VGG_MEAN = [103.939, 116.779, 123.68]


class VGGNet:
    """Builds VGG-16 net structure,
       load parameters from pre-train models.
    """

    def __init__(self, data_dict):
        self.data_dict = data_dict

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name='conv')

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name='fc')

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="bias")

    def conv_layer(self, x, name):
        """Builds convolution layer."""
        with tf.name_scope(name):
            conv_w = self.get_conv_filter(name)
            conv_b = self.get_bias(name)
            h = tf.nn.conv2d(x, conv_w, [1, 1, 1, 1], padding='same')
            h = tf.nn.bias_add(h,conv_b)
            h = tf.nn.relu(h)
            return h

    def pooling_