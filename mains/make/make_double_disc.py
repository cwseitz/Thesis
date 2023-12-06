import numpy as np
import json
import matplotlib.pyplot as plt
from BaseSMLM.generators import *
from skimage.io import imsave
from SPICE.utils import Double

def show_double(adu,double):
    ax[0].imshow(np.sum(adu,axis=0),cmap='gray'); 
    ax[1].imshow(doubled,cmap='gray')
    ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[1].set_xticks([]); ax[1].set_yticks([])
    plt.tight_layout()
    plt.show()

with open('make_double_disc.json', 'r') as f:
    config = json.load(f)
    
stack = []; theta = []; spikes = []
show = True
for n in range(config['ngenerate']):
    print(f'Generating frame {n}')
    disc2d = Ring2D_TwoState(config['nx'],config['ny'])
    nspots = config['nspots']
    args = [config['radius'],nspots]
    fig,ax=plt.subplots(1,2)
    kwargs = config['kwargs']
    adu,_spikes,_theta = disc2d.forward(*args,**kwargs)
    auto,doubled = Double(adu)
    if show:
        show_double(adu,doubled)
    _spikes = _spikes.astype(np.int16)
    stack.append(doubled); spikes.append(_spikes[0])
    theta.append(_theta);
    
stack = np.array(stack); theta = np.array(theta); spikes = np.array(spikes)
imsave(config['savepath']+config['prefix']+'_adu.tif',stack)
imsave(config['savepath']+config['prefix']+'_spikes.tif',spikes)
file = config['savepath']+config['prefix']+'_adu.npz'
np.savez(file,theta=theta)

del stack; del theta; del spikes
