import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread 

#load images
a=imread('images_openpiv/B001_1.tif')
b=imread('images_openpiv/B001_2.tif')

#fig,axs=plt.subplots(1,2,figsize=(12,10))
#axs[0].imshow(a,cmap=plt.cm.gray)
#axs[1].imshow(b,cmap=plt.cm.gray)
#fig.suptitle('Raw images')


#we take interrogation window of 32x32 pixels

win_size=32
a_win=a[:win_size,:win_size].copy()
b_win=b[:win_size,:win_size].copy()

'''fig,axs=plt.subplots(1,2,figsize=(8,8))
axs[0].imshow(a_win,cmap=plt.cm.gray)
axs[1].imshow(b_win,cmap=plt.cm.gray)
fig.suptitle('Interrogation windows')'''


#we can use the cross correlation algorithms but we can also
#do it manually by shifting the first image over the second one

'''fig =plt.imshow(b_win - a_win,cmap=plt.cm.gray)
plt.title('Without shift')
plt.show()'''

#on utilise la fonction roll de numpy pour decalage circulaire

plt.imshow(b_win - np.roll(a_win, (1,0), axis=(0,1)),cmap=plt.cm.gray)
plt.title('Shift down by 1 pixel')
plt.show()

#we are going to find the best shift algorithmically

def match_template(img, template, maxroll=8):
    best-dist=np.inf
    best_shift=(-1,-1):
    for y in range(maxroll):
        for x in range(maxroll):
            #calculate Euclidean distance, on fait la différence avec la distance euclidienne
            dist=np.sqrt(np.sum((img -np.roll(template,(y,x), axis=(0,1)))))
            if dist<best_dist:
                best_dist = dist
                best_shift = (y, x)
    return (best_dist, best_shift)

