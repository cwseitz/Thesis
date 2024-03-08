import tifffile
import numpy as np
import json
from skimage.io import imsave
from glob import glob

""""Decompose datasets for training"""

def stack_to_images(stack,base_name):
    nb = stack.shape[0]
    for n in range(nb):
        imsave(base_name+f'-{n}.tif',np.squeeze(stack[n]))

with open('stack_to_images.json', 'r') as f:
    config = json.load(f)
    
suffix = ['lr','sr','hr']

for sfx in suffix:
    stacks = sorted(glob(config['path']+'*_' + sfx + '.tif'))
    for stack_name in stacks:
        base_name = stack_name.split('.')[0].split('/')[-1]
        stack = tifffile.imread(stack_name)
        stack_to_images(stack,config['savepath']+base_name)


    

