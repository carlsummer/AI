import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
from matplotlib.pyplot import imshow
from PIL import Image

DIR = "./run_style_transfer/"
image_name = "result-00100.jpg"

img = Image.open(os.path.join(DIR,image_name))
print(img.size)

img_arr = np.asarray(img)
imshow(img_arr)
pylab.show()