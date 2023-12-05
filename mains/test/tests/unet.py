import argparse
import collections
import torch
import json
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import napari
from BaseSMLM.generators import Disc2D
from BaseSMLM.psf.psf2d.mix import MLE2DMix
from DeepSMLM.torch.utils import prepare_device
from DeepSMLM.torch.models import LocalizationCNN
from DeepSMLM.torch.train.metrics import jaccard_coeff
from DeepSMLM.localize import UNET_Estimator2D
from scipy.spatial.distance import cdist

class UNET_Test:
    def __init__(self,config):
        self.config=config
    def test(self,plot=True):
        config = self.config
        mix2d = Disc2D(config['nx'],config['ny'])
        adu,spikes,thetagt =\
        mix2d.forward(config['radius'],config['nspots'],N0=config['N0'])
        self.estimator = UNET_Estimator2D(config)
        adu = adu[np.newaxis,np.newaxis,:,:]
        output = self.estimator.forward(adu)
        if plot:
            self.show(adu,output)
            
    def show(self,adu,output):
        fig,ax=plt.subplots(1,2)
        ax[0].imshow(np.squeeze(adu),cmap='gray')
        ax[1].imshow(output[0],cmap='coolwarm')
        plt.show()
