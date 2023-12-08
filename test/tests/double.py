import argparse
import collections
import torch
import json
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import napari
from BaseSMLM.generators import *
from BaseSMLM.psf.psf2d.mix import MLE2DMix
from DeepSMLM.torch.utils import prepare_device
from DeepSMLM.torch.models import LocalizationCNN
from DeepSMLM.torch.train.metrics import jaccard_coeff
from DeepSMLM.localize import NeuralEstimator2D, NeuralEstimatorLoG2D
from SPICE.utils import Double
from scipy.spatial.distance import cdist
from skimage.filters import gaussian

class CNNDouble_Test:
    def __init__(self,config):
        self.config=config
    def test(self,plot=True):
        config = self.config
        mix2d = Disc2D_TwoState(config['nx'],config['ny'])
        args = [config['disc_radius'],config['nspots']]
        kwargs = config['kwargs']
        adu,spikes,thetagt =\
        mix2d.forward(*args,**kwargs)
        self.estimator = NeuralEstimator2D(config)
        auto,doubled = Double(adu)
        doubled = doubled[0]
        smm = np.sum(adu,axis=0)
        doubled = doubled[np.newaxis,np.newaxis,:,:]
        spots,outputs = self.estimator.forward(doubled)
        outputs = gaussian(outputs,sigma=1.0)
        if plot:
            self.show(smm,doubled,outputs,spots,thetagt)
            
    def show(self,smm,doubled,outputs,spots,thetagt):
        fig,ax=plt.subplots(1,3,figsize=(6,2))
        
        ax[1].imshow(np.squeeze(doubled),cmap='gray')
        ax[1].scatter(1.95*thetagt[1,:],1.95*thetagt[0,:],color='red',marker='x')
        ax[1].set_xticks([]); ax[1].set_yticks([])
        #ax[1].legend()
        
        ax[0].imshow(smm,cmap='gray')
        ax[0].scatter(thetagt[1,:],thetagt[0,:],color='red',marker='x')
        ax[0].set_xticks([]); ax[0].set_yticks([])
        
        ax[2].imshow(outputs,cmap='plasma')
        ax[2].scatter(3.9*thetagt[1,:],3.9*thetagt[0,:],color='red',marker='x')
        ax[2].set_xticks([]); ax[2].set_yticks([])
        #ax[2].legend()
        
        plt.tight_layout()
        plt.savefig('/home/cwseitz/Desktop/Doubled-CNN.png',dpi=300)
        plt.show()
