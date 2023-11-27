import numpy as np
import json
import matplotlib.pyplot as plt
from BaseSMLM.generators import Mix2D_Ring
from skimage.io import imsave

with open('make_ring_dataset_2d.json', 'r') as f:
    config = json.load(f)
    
stack = []; theta = []; spikes = []

for n in range(config['ngenerate']):
    print(f'Generating frame {n}')
    mix2d = Mix2D_Ring(config)
    adu,_spikes,_theta = mix2d.generate(plot=False)
    stack.append(adu); theta.append(_theta); spikes.append(_spikes.numpy())
    
stack = np.array(stack); theta = np.array(theta); spikes = np.array(spikes)
imsave(config['savepath']+config['prefix']+'_adu.tif',stack)
imsave(config['savepath']+config['prefix']+'_spikes.tif',spikes)
file = config['savepath']+config['prefix']+'_adu.npz'
np.savez(file,theta=theta)

del stack; del theta; del spikes
