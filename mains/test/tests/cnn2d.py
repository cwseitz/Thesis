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
from DeepSMLM.localize import NeuralEstimator2D
from scipy.spatial.distance import cdist

class CNN2D_Test:
    def __init__(self,config):
        self.modelpath = config['modelpath']
        self.modelname = config['modelname']
        train_config_path = self.modelpath+self.modelname+'/config.json'
        with open(train_config_path,'r') as train_config:
             self.train_config = json.load(train_config)
        self.model,self.device = self.load_model() 
        pixel_size_axial =  2*setup_config['zhrange']/setup_config['nz']
        
    def test(self,plot=True):
        nx,ny = self.cmos_params[2].shape
        lr = self.config['lr']
        mix2d = Mix2D(self.config)
        adu,spikes,thetagt = mix2d.generate(plot=True)
        adu = adu - self.cmos_params[3]
        adu = np.clip(adu,0,None)
        self.estimator = NeuralEstimator2D(self.config)
        spots = self.estimator.forward(adu)
        if plot:
            self.show()

        
    def load_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args = self.train_config['arch']['args']
        model = LocalizationCNN(args['nz'],args['scaling_factor'],dilation_flag=args['dilation_flag'])
        model = model.to(device=device)
        checkpoint = torch.load(self.modelpath+self.modelname+'/'+self.modelname+'.pth', map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model,device
        





  
