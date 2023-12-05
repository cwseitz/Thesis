import numpy as np
import json
import matplotlib.pyplot as plt
from BaseSMLM.generators import Brownian2D, Disc2D
from skimage.io import imsave

"""
This is very similar to the CMOS case, but we just remove readout noise in the config and reduce eta to 0.5. We can do this because mu = lambda*dt. Increasing frame rate therefore, reduces the rate mu, but CNN operates on a time-summed image, so it doesn't matter. It is the same as a CMOS image with gain=1
"""

with open('make_chain_spad_2d.json', 'r') as f:
    config = json.load(f)
    
stack = []; theta = []; spikes = []

for n in range(config['ngenerate']):
    print(f'Generating frame {n}')
    disc2d = Disc2D(config['nx'],config['ny'])
    #nspots = np.random.choice(np.arange(1,6,1))
    nspots = config['nspots']
    args = [config['radius'],nspots]
    #fig,ax=plt.subplots(1,2)
    kwargs = config['kwargs']
    adu,_spikes,_theta = disc2d.forward(*args,**kwargs)
    #ax[0].imshow(adu); ax[1].imshow(_spikes); plt.show()
    _spikes = _spikes.astype(np.int16)
    stack.append(adu); spikes.append(_spikes)
    theta.append(_theta);
    
stack = np.array(stack); theta = np.array(theta); spikes = np.array(spikes)
imsave(config['savepath']+config['prefix']+'_adu.tif',stack)
imsave(config['savepath']+config['prefix']+'_spikes.tif',spikes)
file = config['savepath']+config['prefix']+'_adu.npz'
np.savez(file,theta=theta)

del stack; del theta; del spikes
