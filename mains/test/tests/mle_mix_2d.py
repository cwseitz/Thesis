import numpy as np
import umap
import matplotlib.pyplot as plt
#from BaseSMLM.generators import Mix2D
from BaseSMLM.psf.psf2d import *
from BaseSMLM.psf.psf2d.mix import *

class MLE2DMix_Test:
    """Test a single instance of maximum likelihood estimation"""
    def __init__(self,config):
        self.config = config
        self.cmos_params = [config['eta'],config['texp'],
                            np.load(config['gain'])['arr_0'],
                            np.load(config['offset'])['arr_0'],
                            np.load(config['var'])['arr_0']]  

    def test(self,plot=True):
        nx,ny = self.cmos_params[2].shape
        lr = self.config['lr']
        mix2d = Mix2D(self.config)
        adu,spikes,thetagt = mix2d.generate(plot=True)
        #theta will have shape (ntheta,nspots) as user level
        adu = adu - self.cmos_params[3]
        adu = np.clip(adu,0,None)
        theta0 = np.zeros_like(thetagt); theta0 += thetagt
        #theta0 += np.random.normal(0,1,size=theta0.shape) #add parameter noise
        pipe_cnn = PipelineCNN2D(config,dataset)
        spots_cnn = pipe_cnn.localize()
        if plot:
            self.show(adu,loglike,thetat,theta,theta0,thetagt)
            
                        
    def show(self,adu,loglike,thetat,theta,theta0,thetagt):
        fig,ax=plt.subplots()
        ax.imshow(adu,cmap='gray')
        ax.scatter(theta[1,:],theta[0,:],marker='x',s=5,color='red')
        ax.scatter(theta0[1,:],theta0[0,:],marker='x',s=5,color='blue')
        plt.show()

