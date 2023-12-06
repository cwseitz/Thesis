import argparse
import collections
import torch
import json
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import napari
from BaseSMLM.generators import Disc2D
from DeepSMLM.torch.utils import prepare_device
from DeepSMLM.torch.models import LocalizationVAE2

class VAE2_Test:
    def __init__(self,config):
        self.config=config
        self.modelpath = config['modelpath']
        self.modelname = config['modelname']
        self.model,self.device = self.load_model()

        
    def load_model(self):
        config = self.config
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_config = self.modelpath+self.modelname+'/'+self.modelname+'.json'
        with open(train_config, 'r') as f:
            train_config = json.load(f)
        args = train_config['arch']['args']
        model = LocalizationVAE2(args['latent_dim'],args['nx'],args['ny'])
        model = model.to(device=device)
        checkpoint = self.modelpath+self.modelname+'/'+self.modelname+'.pth'
        checkpoint = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model,device

    def test(self,plot=True):
        config = self.config
        mix2d = Disc2D(config['nx'],config['ny'])
        adu,spikes,thetagt =\
        mix2d.forward(config['radius'],config['nspots'],N0=config['N0'],offset=0.0,var=0.0)
        adu = adu[np.newaxis,np.newaxis,:,:]
        adut = torch.tensor(adu,dtype=torch.float32).cuda()
        pred,conv,mu,logvar = self.model(adut)
        conv = conv.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        mu = mu.cpu().detach().numpy()
        mu = mu.reshape((2,config['nspots']))
        self.show(adu,pred,conv,mu,thetagt)

    def show(self,adu,pred,conv,mu,thetagt):
        mu = mu + self.config['nx']/2
        fig,ax=plt.subplots(1,3)
        ax[0].imshow(np.squeeze(adu),cmap='gray')
        ax[1].imshow(np.squeeze(pred),cmap='gray')
        ax[0].scatter(thetagt[1,:],thetagt[0,:],marker='x',
                   color='blue',s=5,label='True')
        ax[1].scatter(thetagt[1,:],thetagt[0,:],marker='x',
                   color='blue',s=5,label='True')
        ax[1].scatter(mu[1,:],mu[0,:],marker='x',
                   color='red',s=5,label='Pred')
        ax[2].imshow(np.squeeze(conv),cmap='plasma')
        plt.tight_layout()
        plt.show()

