from openpiv import tools, pyprocess, validation, filters, scaling

import numpy as np
import matplotlib.pyplot as plt
import imageio
import importlib_resources
import pathlib

path = importlib_resources.files('openpiv')

frame_a=tools.imread('images_openpiv/B001_1.tif')
frame_b=tools.imread('images_openpiv/B001_2.tif')

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

#3 arrays returned : u component of the velocity vector /
#v component of the velocity vector / the s2n ratio

x, y = pyprocess.get_coordinates(
    image_size=frame_a.shape,
    search_area_size=searchsize,
    overlap=overlap
)

#POST PROCESSING
#sig2noise_val : get a mask to indicate which vectors have a minimum amount
#oh s2n. Vectors below a certain threshol substituted by NaN

invalid_mask=validation.sig2noise_val (
    sig2noise,
    threshold=1.05,
)

#replace_outliers : to find outlier vectors and substitute them by an 
#average of neighbouring vectors
#the larger the kernel size, the larger the neighbourhood considered
#the amount of iterations can be chosen via max_iter

u2, v2 = filters.replace_outliers(
    u0, v0,
    invalid_mask,
    method='localmean',
    max_iter=3,
    kernel_size=3,
)

# convert x,y to mm
# convert u,v to mm/sec

x, y, u3, v3 = scaling.uniform(
    x, y, u2, v2,
    scaling_factor = 96.52,  # 96.52 pixels/millimeter
)

# 0,0 shall be bottom left, positive rotation rate is counterclockwise
x, y, u3, v3 = tools.transform_coordinates(x, y, u3, v3)

#RESULTS

tools.save('exp1_001.txt' , x, y, u3, v3, invalid_mask)

fig, ax = plt.subplots(figsize=(8,8))
tools.display_vector_field(
    pathlib.Path('exp1_001.txt'),
    ax=ax, scaling_factor=96.52,
    scale=50, # scale defines here the arrow length
    width=0.0035, # width is the thickness of the arrow
    on_img=True, # overlay on the image
    image_name= str(path / 'data'/'test1'/'exp1_001_a.bmp'),
);