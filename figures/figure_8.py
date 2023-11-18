import numpy as np
import pandas as pd
import ripleyk
import matplotlib.pyplot as plt
from SMLM.utils import *

class Figure_8:
    """Density Figure"""
    def __init__(self, config):
        self.config = config
        
    def add_interval(self,spots,interval_size=10):
        max_frame = spots['frame'].max()
        intervals = range(0, max_frame + interval_size, interval_size)
        spots['interval'] = pd.cut(spots['frame'], bins=intervals, right=False, labels=intervals[:-1])
        return spots
        
    def plot(self, prefixes, pixel_size=108.3, nbins=500):  
    
    
        spots = pd.read_csv(self.config['analpath'] + prefixes[0] + '.csv')
        spots = self.add_interval(spots,interval_size=100)
        intervals = spots['interval'].unique()
        fig,ax=plt.subplots(1,len(intervals)+1,figsize=(12,4),sharex=True,sharey=True) 
        H_all = np.zeros((nbins,nbins))
        for n,interval in enumerate(intervals):
            this_spots = spots.loc[spots['interval'] == interval]
            H,xedges,yedges = np.histogram2d(this_spots['x_mle'], this_spots['y_mle'],bins=nbins)
            H_all += H
            x_range = (spots['x_mle'].max()-spots['x_mle'].min())*0.1083
            y_range = (spots['y_mle'].max()-spots['y_mle'].min())*0.1083
            x_bin_size = x_range/nbins
            y_bin_size = y_range/nbins
            print(f'X bin size: {x_bin_size} um')
            print(f'Y bin size: {y_bin_size} um')
            ax[n].imshow(H,cmap='gray',vmin=0.0,vmax=2.0,aspect=x_bin_size/y_bin_size)
            ax[n].set_xticks([]); ax[n].set_yticks([])
            ax[n].invert_yaxis()
        ax[-1].imshow(H_all,cmap='gray',vmin=0.0,vmax=5.0,aspect=x_bin_size/y_bin_size)
        ax[-1].set_xticks([]); ax[n].set_yticks([])
        ax[-1].invert_yaxis()
        plt.tight_layout()
        plt.show()
        



        

