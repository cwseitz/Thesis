import numpy as np
import matplotlib.pyplot as plt
import tifffile
import pandas as pd
from matplotlib.colors import *
from SMLM.utils import KDE
from SMLM.generators import Mix2D

class Figure_9:
    """Time-resolved STORM concept figure"""
    def __init__(self,config):
        self.config = config

    def plot1(self,prefix):

        stack = tifffile.imread(self.config['datapath']+prefix+'-sub.tif')
        
        spots = pd.read_csv(self.config['analpath'] + prefix + '/' + prefix + '_spots.csv')
        spots = spots.loc[spots['peak'] > 200]
        spots = spots.sample(20000)
        render = KDE(spots).get_kde()
        
        cvals  = [-2., 2]
        colors = ["black","red"]
        norm=plt.Normalize(min(cvals),max(cvals))
        tuples = list(zip(map(norm,cvals), colors))
        cmap = LinearSegmentedColormap.from_list("", tuples)        

        fig,ax=plt.subplots()
        ax.imshow(stack[0,:,:],cmap='gray',vmin=0,vmax=200.0)
        ax.set_xticks([]); ax.set_yticks([])
        
        fig,ax=plt.subplots()
        ax.imshow(stack[:,90,:].T,cmap='gray',vmin=0,vmax=200.0)
        ax.set_aspect(3.0)
        ax.set_xticks([]); ax.set_yticks([])    
        
        fig,ax=plt.subplots()
        render = render/render.max()
        ax.imshow(render,cmap='gray',vmin=0,vmax=0.25)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect(1.0)
        
        fig,ax=plt.subplots()
        ax.scatter(spots['y_mle'],spots['x_mle'],s=1,color='black')
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect(1.0)
        
    def plot2(self):
        fig,ax=plt.subplots(1,3,sharex=True,sharey=True)
        ks = [1,2,5]
        for nk, k in enumerate(ks):
            config = self.config.copy()
            config['particles'] = k
            mix2d = Mix2D(config)
            adu,_spikes,_theta = mix2d.generate(r=2)
            ax[nk].imshow(adu[0],cmap='gray')
            center = (11.5,11.5)
            dashed_circle = plt.Circle(center, 2.56, fill=False, color='yellow', linestyle='dashed', linewidth=2)
            ax[nk].add_patch(dashed_circle)
            ax[nk].set_xticks([]); ax[nk].set_yticks([])
