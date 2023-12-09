from BaseSMLM.generators import *
from SPICE.utils import Double
from skimage.io import imread,imsave
import matplotlib.pyplot as plt
import numpy as np

radius = 5
nspots = 5
nx = ny = 20

disc2d = Disc2D_TwoState(nx,ny)
adu,spikes,theta = disc2d.forward(radius,nspots,N0=10.0,B0=0.2,offset=0.0,var=0.0,nframes=100,show=False)
auto,doubled = Double(adu)
imsave('/home/cwseitz/Desktop/adu.tif',adu)
imsave('/home/cwseitz/Desktop/doubled.tif',doubled)

"""
stack = imread('/home/cwseitz/Desktop/doubled.tif')
nt,nx,ny = stack.shape
stack = stack.reshape((nt,nx*ny))
fig,ax=plt.subplots()
for n in range(nx*ny):
    ax.plot(stack[:,n])
plt.show()
"""

"""
fig,ax=plt.subplots(1,3)
ax[0].imshow(np.sum(adu,axis=0),cmap='gray')
ax[2].imshow(doubled,cmap='plasma')
ax[0].scatter(theta[1,:],theta[0,:],color='red',marker='x')
ax[1].imshow(auto,cmap='gray')
ax[2].scatter(2*theta[1,:],2*theta[0,:],color='blue',marker='x')
plt.show()
"""
