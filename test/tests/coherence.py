import argparse
import collections
import torch
import json
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import napari
from BaseSMLM.generators import *
from DeepSMLM.torch.utils import prepare_device
from DeepSMLM.torch.models import LocalizationCNN
from DeepSMLM.torch.train.metrics import jaccard_coeff
from DeepSMLM.localize import NeuralEstimator2D, NeuralEstimatorLoG2D
from oci.utils import G2
from scipy.spatial.distance import cdist
from skimage.filters import gaussian

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

class CNNCoherence_Test:
    def __init__(self,config):
        self.config=config
    def test(self,plot=True):
        config = self.config
        disc2d = GaussianRing2D_TwoState(config['nx'],config['ny'])
        args = [config['disc_radius'],config['nspots']]
        kwargs = config['kwargs']
        adu,spikes,thetagt =\
        disc2d.forward(*args,**kwargs,show=False)
        summed = np.sum(adu,axis=0)
        self.estimator = NeuralEstimator2D(config)
        g2 = linear_interpolation(G2(adu))
        g2 = g2[np.newaxis,np.newaxis,:,:]
        spots,outputs = self.estimator.forward(g2)
        outputs = gaussian(outputs,sigma=1.0)
        if plot:
            self.show(summed,g2,outputs,spots,thetagt)
            
    def show(self,summed,doubled,outputs,spots,thetagt):
        fig,ax=plt.subplots(1,3,figsize=(6,2))

        ax[0].scatter(thetagt[1,:],thetagt[0,:],color='red',marker='x')  
        ax[0].invert_yaxis()      
        ax[0].imshow(summed,cmap='gray')
        ax[0].set_xticks([]); ax[0].set_yticks([])
        
        ax[1].imshow(np.squeeze(doubled),cmap='gray')
        ax[1].scatter(1.95*thetagt[1,:],1.95*thetagt[0,:],color='red',marker='x')
        ax[1].set_xticks([]); ax[1].set_yticks([])
        ax[1].legend()
                
        ax[2].imshow(outputs,cmap='plasma')
        ax[2].scatter(3.9*thetagt[1,:],3.9*thetagt[0,:],color='red',marker='x')
        ax[2].set_xticks([]); ax[2].set_yticks([])
        ax[2].legend()

        plt.show()
