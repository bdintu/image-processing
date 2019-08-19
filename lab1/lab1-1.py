import cv2
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.image as mpimg
from matplotlib.ticker import LinearLocator, FormatStrFormatter


img_path = 'stinkbug.png'

img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print('shape', img_gray.shape)
print('dtype', img_gray.dtype)

f1 = plt.figure('Figure 1')
plt.subplot(1, 2, 1), plt.imshow(img_rgb)
plt.subplot(1, 2, 2), plt.imshow(img_gray, 'gray')

f2 = plt.figure('Figure 2')
plt.subplot(2, 2, 1), plt.imshow(img)
plt.subplot(2, 2, 2), plt.imshow(img_gray, 'gray')
plt.subplot(2, 2, 3), plt.imshow(img)
plt.subplot(2, 2, 4), plt.imshow(img_gray, 'gray')

f3 = plt.figure('Figure 3')
plt.imshow(img)


plt.show()


#cv2.imshow('image',img_gray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
