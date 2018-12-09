import tensorflow as tf

hello = tf.constant("hello tf!")
sess = tf.Session()
print(sess.run(hello))

import cv2
print('hello opencv')
