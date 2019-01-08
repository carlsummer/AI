# resize
# crop
# flip
# brightness & contrast
# tf.image.resize_area
# tf.image.resize_bicubic
# tf.image.resize_nearest_neightbor

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import pylab

name = "./gugong.jpg"
img_string = tf.read_file(name)
img_decoded = tf.image.decode_image(img_string)
img_decoded = tf.reshape(img_decoded,[1,649,1018,3])

resize_img = tf.image.resize_bicubic(img_decoded,[1298,2036])

sess = tf.Session()
img_decoded_val = sess.run(resize_img)
img_decoded_val = img_decoded_val.reshape((1298,2036,3))
img_decoded_val = np.asarray(img_decoded_val,np.uint8)
print(img_decoded_val.shape)

imshow(img_decoded_val)
pylab.show()