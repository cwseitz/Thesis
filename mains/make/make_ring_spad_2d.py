import numpy as np
import json
import matplotlib.pyplot as plt
from BaseSMLM.generators import Ring2D
from skimage.io import imsave

"""
This is very similar to the CMOS case, but we just remove readout noise in the config and reduce eta to 0.5. We can do this because mu = lambda*dt. Increasing frame rate therefore, reduces the rate mu, but CNN operates on a time-summed image, so it doesn't matter. It is the same as a CMOS image with gain=1
"""

with open('make_ring_spad_2d.json', 'r') as f:
    config = json.load(f)
    
stack = []; theta = []; spikes = []

for n in range(config['ngenerate']):
    print(f'Generating frame {n}')
    ring2d = Ring2D(config['nx'],config['ny'])
    args = [config['radius'],config['nspots']]
    kwargs = config['kwargs']
    adu,_spikes,_theta = ring2d.forward(*args,**kwargs)
    _spikes = _spikes.astype(np.int16)
    stack.append(adu); theta.append(_theta); spikes.append(_spikes)
    
stack = np.array(stack); theta = np.array(theta); spikes = np.array(spikes)
imsave(config['savepath']+config['prefix']+'_adu.tif',stack)
imsave(config['savepath']+config['prefix']+'_spikes.tif',spikes)
file = config['savepath']+config['prefix']+'_adu.npz'
np.savez(file,theta=theta)

del stack; del theta; del spikes
