import numpy as np
import matplotlib.pyplot as plt
from BaseSMLM.generators import Iso2D
from BaseSMLM.psf.psf2d import *
from .mcmc import *

class MLE2DIso_Test:
    """Test a single instance of maximum likelihood estimation"""
    def __init__(self,config):
        self.config = config
        self.cmos_params = [config['eta'],config['texp'],
                            np.load(config['gain'])['arr_0'],
                            np.load(config['offset'])['arr_0'],
                            np.load(config['var'])['arr_0']]  
    def test(self):
        gain,offset,var = self.cmos_params[2:]
        nx,ny = offset.shape
        x0 = nx // 2; y0 = ny // 2
        self.thetagt = np.array([x0,y0,self.config['sigma'],
                                 self.config['N0']])
        iso2d = Iso2D(self.thetagt,self.config)
        adu = iso2d.generate(plot=True)
        adu = adu - offset
        adu = np.clip(adu,0,None)
        lr = np.array([0.001,0.001,0.0,100.0]) #hyperpar
        theta0 = np.zeros_like(self.thetagt)
        theta0 += self.thetagt
        theta0[0] += np.random.normal(0,1)
        theta0[1] += np.random.normal(0,1)
        theta0[3] += 100
        opt = MLE2D(theta0,adu,self.config,theta_gt=self.thetagt)
        theta,loglike,conv = opt.optimize(max_iters=100,lr=lr,plot_fit=True)
        run_mcmc(theta,adu,self.cmos_params,num_samples=1000,warmup_steps=200)



       
