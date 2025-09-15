from openpiv import tools, pyprocess, validation, filters, scaling

import numpy as np
import matplotlib.pyplot as plt
import imageio
import importlib_resources
import pathlib

path = importlib_resources.files('openpiv')

frame_a=tools.imread('images_openpiv/B005_1.tif')
frame_b=tools.imread('images_openpiv/B005_2.tif')

fig,ax = plt.subplots(1,2,figsize=(12,10))
ax[0].imshow(frame_a,cmap=plt.cm.gray)
ax[1].imshow(frame_b,cmap=plt.cm.gray)
#plt.show()

#PROCESSING
#extended_search_area_piv : standard PIV cross-correlation algorithm
#allows the search area (search_area_size) in the 2nd frame to be larger
#than the interrogation window in the 1st frame (window_size)
#the search areas can overlap

winsize = 32 # pixels, interrogation window size in frame A
searchsize = 38  # pixels, search area size in frame B
overlap = 17 # pixels, 50% overlap
dt = 0.02 # sec, time interval between the two frames

u0, v0, sig2noise = pyprocess.extended_search_area_piv(
    frame_a.astype(np.int32),
    frame_b.astype(np.int32),
    window_size=winsize,
    overlap=overlap,
    dt=dt,
    search_area_size=searchsize,
    sig2noise_method='peak2peak',
)

