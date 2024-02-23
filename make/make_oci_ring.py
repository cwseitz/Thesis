import numpy as np
import json
import matplotlib.pyplot as plt
from BaseSMLM.generators import *
from skimage.io import imsave
from oci.utils import G2

def linear_interpolation(image):
    rows, cols = image.shape
    interpolated_image = np.zeros_like(image)
    nx,ny = image.shape
    zero_pixels = np.argwhere(image == 0)
    for row, col in zero_pixels:
        if row > 0 and row < nx-1 and col > 0 and col < ny-1:
            image[row, col] = (image[row+1, col]+image[row-1, col]+image[row, col+1]+image[row, col-1])/4
        else:
            image[row,col] = 1.0
    return image

with open('make_oci_ring.json', 'r') as f:
    config = json.load(f)
    
stack = []; theta = []; spikes = []; summed = []
show = False
for n in range(config['ngenerate']):
    print(f'Generating frame {n}')
    ring2d = GaussianRing2D_TwoState(config['nx'],config['ny'])
    nspots = config['nspots']
    args = [config['radius'],nspots]
    fig,ax=plt.subplots(1,2)
    kwargs = config['kwargs']
    adu,_spikes,_theta = ring2d.forward(*args,**kwargs)
    ssum = np.sum(adu,axis=0)
    g2 = G2(adu)
    g2 = linear_interpolation(g2)
    if show:
        fig,ax=plt.subplots(1,2)
        ax[0].imshow(ssum); ax[1].imshow(g2)
        plt.show()
    _spikes = _spikes.astype(np.int16)
    stack.append(g2); spikes.append(_spikes[0])
    theta.append(_theta); summed.append(ssum)
    
stack = np.array(stack); theta = np.array(theta); spikes = np.array(spikes)
summed = np.array(summed)
imsave(config['savepath']+config['prefix']+'_adu.tif',stack)
imsave(config['savepath']+config['prefix']+'_spikes.tif',spikes)
file = config['savepath']+config['prefix']+'.npz'
np.savez(file,theta=theta)

del stack; del theta; del spikes
