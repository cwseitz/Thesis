import argparse
import collections
import torch
import json
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import napari
from BaseSMLM.generators import Ring2D
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
        cnn_path = '/home/cwseitz/Desktop/Torch/Models2D/DeepSTORM/'
        cnn_name = '1205_040712'
        model.load_cnn(cnn_path,cnn_name)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model,device

    def test(self,plot=True):
        config = self.config
        mix2d = Ring2D(config['nx'],config['ny'])
        adu,spikes,thetagt =\
        mix2d.forward(config['radius'],config['nspots'],N0=config['N0'],offset=0.0,var=0.0)
        adu = adu[np.newaxis,np.newaxis,:,:]
        adut = torch.tensor(adu,dtype=torch.float32).cuda()
        pred,mu,logvar = self.model(adut)
        pred = pred.cpu().detach().numpy()
        mu = mu.cpu().detach().numpy()
        mu = mu.reshape((2,config['nspots']))
        self.show(adu,pred,mu,thetagt)

    def show(self,adu,pred,mu,thetagt):
        mu = mu + self.config['nx']/2
        fig,ax=plt.subplots(1,2)
        ax[0].imshow(np.squeeze(adu),cmap='gray')
        ax[1].imshow(np.squeeze(pred),cmap='gray')
        ax[0].scatter(thetagt[1,:],thetagt[0,:],marker='x',
                   color='blue',s=5,label='True')
        ax[1].scatter(thetagt[1,:],thetagt[0,:],marker='x',
                   color='blue',s=5,label='True')
        ax[1].scatter(mu[1,:],mu[0,:],marker='x',
                   color='red',s=5,label='Pred')
        plt.tight_layout()
        plt.show()

