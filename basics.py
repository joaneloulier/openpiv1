import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread 

#load images
a=imread('images_openpiv/B001_1.tif')
b=imread('images_openpiv/B001_2.tif')

fig,axs=plt.subplots(1,2,figsize=(12,10))
axs[0].imshow(a,cmap=plt.cm.gray)
axs[1].imshow(b,cmap=plt.cm.gray)
fig.suptitle('Raw images')
plt.show()


