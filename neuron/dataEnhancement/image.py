# resize
# crop
# flip
# brightness & contrast
# tf.image.resize_area
# tf.image.resize_bicubic
# tf.image.resize_nearest_neightbor
"""
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
"""

# crop
# tf.image.pad_to_bounding_box 填充
# tf.image.crop_to_bounding_box 裁剪
# tf.random_crop
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import pylab

name = "./gugong.jpg"
img_string = tf.read_file(name)
img_decoded = tf.image.decode_image(img_string)
img_decoded = tf.reshape(img_decoded,[1,649,1018,3])

padded_img = tf.image.pad_to_bounding_box(img_decoded,0,0,749,1118)

sess = tf.Session()
img_decoded_val = sess.run(padded_img)
img_decoded_val = img_decoded_val.reshape((749,1118,3))
img_decoded_val = np.asarray(img_decoded_val,np.uint8)
print(img_decoded_val.shape)

imshow(img_decoded_val)
pylab.show()
"""

# flip
# tf.image.flip_up_down
# tf.image.flip_left_right
# tf.image.flip_up_down
# tf.image.flip_left_right
"""
import numpy as np
import tensorflow as tf
from matplotlib.pyplot import imshow
import pylab

name = "./gugong.jpg"
img_string = tf.read_file(name)
img_decoded = tf.image.decode_image(img_string)
img_decoded = tf.reshape(img_decoded,[1,649,1018,3])

flipped_img = tf.image.flip_up_down(img_decoded)

sess = tf.Session()
img_decoded_val = sess.run(flipped_img)
img_decoded_val = img_decoded_val.reshape((649,1018,3))
img_decoded_val = np.asarray(img_decoded_val,np.uint8)
print(img_decoded_val.shape)

imshow(img_decoded_val)
pylab.show()
"""

# brightness
# tf.image.adjust_brightness
# tf.image.random_brightness
# tf.image.adjust_constrast
# tf.image.random_constrast
import numpy as np
import tensorflow as tf
from matplotlib.pyplot import imshow
import pylab

name = "./gugong.jpg"
img_string = tf.read_file(name)
img_decoded = tf.image.decode_image(img_string)
img_decoded = tf.reshape(img_decoded,[1,649,1018,3])

new_img = tf.image.adjust_brightness(img_decoded, -0.5)

sess = tf.Session()
img_decoded_val = sess.run(new_img)
img_decoded_val = img_decoded_val.reshape((649,1018,3))
img_decoded_val = np.asarray(img_decoded_val,np.uint8)
print(img_decoded_val.shape)

imshow(img_decoded_val)
pylab.show()
