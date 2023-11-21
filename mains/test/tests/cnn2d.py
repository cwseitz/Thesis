import argparse
import collections
import torch
import json
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import napari
from BaseSMLM.generators import Mix2D
from DeepSMLM.torch.utils import prepare_device
from DeepSMLM.torch.models import LocalizationCNN
from DeepSMLM.torch.train.metrics import jaccard_coeff
from DeepSMLM.localize import NeuralEstimator2D, NeuralEstimatorLoG2D
from scipy.spatial.distance import cdist

class CNN2D_Test:
    def __init__(self,config):
        self.modelpath = config['modelpath']
        self.modelname = config['modelname']
        self.config = config
        train_config_path = self.modelpath+self.modelname+'/config.json'
        with open(train_config_path,'r') as train_config:
             self.train_config = json.load(train_config)
        self.model,self.device = self.load_model() 
        
    def test(self,plot=True):
        mix2d = Mix2D(self.config)
        adu,spikes,thetagt = mix2d.generate(plot=True)
        self.estimator = NeuralEstimatorLoG2D(self.config)
        adu = adu[np.newaxis,np.newaxis,:,:]
        spots,outputs = self.estimator.forward(adu)
        if plot:
            self.show(adu,outputs,spots,thetagt)
            
    def show(self,adu,outputs,spots,thetagt):
        fig,ax=plt.subplots(1,2)
        ax[0].imshow(np.squeeze(adu),cmap='gray')
        ax[0].scatter(spots['y']/4,spots['x']/4,marker='x',
                   color='red',s=5,label='CNN')
        ax[0].scatter(thetagt[1,:],thetagt[0,:],marker='x',
                   color='blue',s=5,label='True')
        ax[0].legend()
        ax[1].imshow(outputs,cmap='plasma')
        ax[1].scatter(spots['y'],spots['x'],marker='x',
                   color='red',s=5,label='CNN')
        ax[1].scatter(4*thetagt[1,:],4*thetagt[0,:],marker='x',
                   color='cyan',s=5,label='True')
        ax[1].legend()
        plt.tight_layout()
        plt.show()

    def load_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args = self.train_config['arch']['args']
        model = LocalizationCNN(args['nz'],args['scaling_factor'],dilation_flag=args['dilation_flag'])
        model = model.to(device=device)
        checkpoint = torch.load(self.modelpath+self.modelname+'/'+self.modelname+'.pth', map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model,device
        





  
