import numpy as np
import matplotlib.pyplot as plt
from BaseSMLM.generators import Mix2D
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

    def test(self):
        nx,ny = self.cmos_params[2].shape
        lr = self.config['lr']
        mix2d = Mix2D(self.config)
        adu,spikes,theta = mix2d.generate(plot=True)
        adu = adu - self.cmos_params[3]
        adu = np.clip(adu,0,None)
        theta0 = np.zeros_like(theta); theta0 += theta
        theta0 += np.random.normal(0,1,size=theta0.shape)
        opt = MLE2DMix(theta0,adu,self.config,theta_gt=theta)
        theta, loglike = opt.optimize(max_iters=100,lr=lr,plot_fit=True)

