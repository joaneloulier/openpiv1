import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread 

#load images
a=imread('images_openpiv/B005_1.tif')
b=imread('images_openpiv/B005_2.tif')

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

'''plt.imshow(b_win - np.roll(a_win, (1,0), axis=(0,1)),cmap=plt.cm.gray)
plt.title('Shift down by 1 pixel')
plt.show()'''

#we are going to find the best shift algorithmically

def match_template(img, template, maxroll=8):
    best_dist=np.inf
    best_shift=(-1,-1)
    for y in range(maxroll):
        for x in range(maxroll):
            #calculate Euclidean distance, on fait la différence avec la distance euclidienne
            dist=np.sqrt(np.sum((img -np.roll(template,(y,x), axis=(0,1)))))
            if dist<best_dist:
                best_dist = dist
                best_shift = (y, x)
    return (best_dist, best_shift)

# indeed, when we find the correct shift, we got zero distance. it's not so in real images:
best_dist, best_shift = match_template(b_win, a_win)
print(f"{best_dist=}")
print(f"{best_shift=}")

#on va dessiner les vecteurs de vélocité

#u=delta(x pixels)/delta(t) v=etc..

#compliqué car maintenant on a besoin de faire le même calcul
#chaque portion de 32*32 afin de couvrir toute l'image
#on va donc faire un procédé de cross_correlation:

from scipy.signal import correlate

cross_corr = correlate(b_win - b_win.mean(), a_win - a_win.mean(), method="fft")
# Note that it's approximately twice as large than the original windows, as we
# can shift a_win by a maximum of it's size - 1 horizontally and vertically
# while still maintaining some overlap between the two windows.
print("Size of the correlation map: %d x %d" % cross_corr.shape)

# let's see what the cross-correlation looks like
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
Y, X = np.meshgrid(np.arange(cross_corr.shape[0]), np.arange(cross_corr.shape[1]))

ax.plot_surface(Y, X, cross_corr, cmap='jet', linewidth=0.2)  # type: ignore
plt.title("B005 : Correlation map — peak is the most probable shift, for one interrogation window")
fig.savefig("C:/Users/JL284743/Documents/github/resultats/openPiv1/basics/correlation_map_B005.png")
plt.show()

# let's see the same correlation map, from above
#plt.imshow(cross_corr, cmap=plt.cm.gray)

y, x = np.unravel_index(cross_corr.argmax(), cross_corr.shape)
print(f"{y=}, {x=}")

#plt.plot(x, y, "ro")
#plt.show()

dy, dx = y - 31, x - 31
print(f"{dy=}, {dx=}")

#pour toutes les petites fenêtres nous allons répéter l'analyse
#on prend des fenêtres de 32*32 et on fait le travail en boucle
def vel_field(curr_frame, next_frame, win_size):
    ys = np.arange(0, curr_frame.shape[0], win_size)
    xs = np.arange(0, curr_frame.shape[1], win_size)
    dys = np.zeros((len(ys), len(xs)))
    dxs = np.zeros((len(ys), len(xs)))
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            int_win = curr_frame[y : y + win_size, x : x + win_size]
            search_win = next_frame[y : y + win_size, x : x + win_size]
            cross_corr = correlate(
                search_win - search_win.mean(), int_win - int_win.mean(), method="fft"
            )
            dys[iy, ix], dxs[iy, ix] = (
                np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
                - np.array([win_size, win_size])
                + 1
            )
    # draw velocity vectors from the center of each window
    ys = ys + win_size / 2
    xs = xs + win_size / 2
    return xs, ys, dxs, dys

xs, ys, dxs, dys = vel_field(a, b, 32)
norm_drs = np.sqrt(dxs ** 2 + dys ** 2)

fig, ax = plt.subplots(figsize=(6, 6))
# we need these flips on y since quiver uses a bottom-left origin, while our
# arrays use a top-right origin
ax.quiver(
    xs,
    ys[::-1],
    dxs,
    -dys,
    norm_drs,
    cmap="plasma",
    angles="xy",
    scale_units="xy",
    scale=0.25,
)
ax.set_aspect("equal")
plt.title('Champ des vitesses : B005')
fig.savefig("C:/Users/JL284743/Documents/github/resultats/openPiv1/basics/velocity_field_B005.png")
plt.show()
