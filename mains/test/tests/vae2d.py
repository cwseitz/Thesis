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
from DeepSMLM.torch.models import LocalizationVAE

class VAE2D_Test:
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
        model = LocalizationVAE(args['latent_dim'],args['nx'],args['ny'])
        model = model.to(device=device)
        checkpoint = self.modelpath+self.modelname+'/'+self.modelname+'.pth'
        checkpoint = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model,device

    def test(self,plot=True):
        config = self.config
        mix2d = Ring2D(config['nx'],config['ny'])
        adu,spikes,thetagt =\
        mix2d.forward(config['radius'],config['nspots'],N0=config['N0'])
        adu = adu[np.newaxis,np.newaxis,:,:]
        adut = torch.tensor(adu,dtype=torch.float32).cuda()
        pred,mu,logvar = self.model(adut)
        pred = pred.cpu().detach().numpy()
        self.show(adu,pred)

    def show(self,adu,pred):
        fig,ax=plt.subplots(1,2)
        ax[0].imshow(np.squeeze(adu),cmap='gray')
        ax[1].imshow(np.squeeze(pred),cmap='gray')
        #ax.scatter(thetagt[1,:],thetagt[0,:],marker='x',
        #           color='blue',s=5,label='True')
        #ax.scatter(theta[1,:],theta[0,:],marker='x',
        #           color='red',s=5,label='Pred')
        plt.tight_layout()
        plt.show()

