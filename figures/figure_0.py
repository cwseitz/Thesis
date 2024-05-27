import numpy as np
import matplotlib.pyplot as plt
from .kdist import *
from skimage.io import imread

class Figure_0:
    """Noise model and quality of Poisson approximation to convolution distribution"""
    
    def __init__(self,config,prefix):
        self.config = config
        self.prefix = prefix
        self.stack = imread(config['datapath'] + prefix + '.tif')
        
    def plot(self):
        fig, ax = plt.subplots(2,3,figsize=(9,5))
        self.add_cmfs(ax[0,2])
        self.add_kdist_var(ax[1,2])
        self.add_noise_maps(ax)
        plt.tight_layout()
        
    def add_cmfs(self,ax):
        offset = 100
        std = np.sqrt(5)
        rate = 500
        get_kdist(offset,rate,std,ax=ax)
        ax.set_ylabel('Cumulative probability')
        ax.set_xlabel('ADU')
        ax.set_title(r'$\mu_{k}$ = 500')
        
    def add_kdist_var(self,ax):
        offset = 100
        std = np.sqrt(5)
        rate_space = np.linspace(100,5000,100) #mu
        dist_space = np.zeros_like(rate_space)
        for i,this_rate in enumerate(rate_space):
            dist_space[i] = get_kdist(offset,this_rate,std)
        ax.plot(rate_space,dist_space,color='gray')
        ax.set_xlabel(r'$\mu_{k}$')
        ax.set_ylabel('Komogonov distance')

    def add_noise_maps(self,ax):

        var_map = np.var(self.stack,axis=0)
        avg_map = np.mean(self.stack,axis=0)

        im1 = ax[0,0].imshow(avg_map,vmin=95,vmax=105,cmap='coolwarm')
        ax[0,0].set_title('CMOS Offset')
        ax[0,0].set_xlabel('Pixels'); ax[0,0].set_ylabel('Pixels')
        ax[0,0].set_xticks([0,1000]); ax[0,0].set_yticks([0,1000])
        im2 = ax[0,1].imshow(var_map,vmin=0,vmax=20,cmap='coolwarm')
        ax[0,1].set_xticks([0,1000]); ax[0,1].set_yticks([0,1000])
        ax[0,1].set_title('CMOS Variance')
        ax[0,1].set_xlabel('Pixels'); ax[0,1].set_ylabel('Pixels')

        plt.colorbar(im1, ax=ax[0,0], label=r'$ADU$')
        plt.colorbar(im2, ax=ax[0,1], label=r'$ADU^{2}$')

        bins = np.linspace(95,105,100)
        vals,bins = np.histogram(avg_map.flatten(),bins=bins,density=True)
        ax[1,0].plot(bins[:-1],vals,marker='o',markersize=5,color='black',linestyle='')
        ax[1,0].set_xlabel(r'Offset ($ADU$)')
        bins = np.linspace(0,20,100)
        vals,bins = np.histogram(var_map.flatten(),bins=bins,density=True)
        ax[1,1].plot(bins[:-1],vals,marker='o',markersize=5,color='black',linestyle='')
        bins = np.linspace(95,105,100)
        ax[1,1].set_xlabel(r'Variance ($ADU^{2}$)')

