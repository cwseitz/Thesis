import numpy as np
import matplotlib.pyplot as plt
from BaseSMLM.generators import Iso3D
from BaseSMLM.psf.psf3d import *

class MLE3D_Test:
    """Test a single instance of MLE for 3D psf"""
    def __init__(self,config):
        self.config = config
        self.cmos_params = [config['eta'],config['texp'],
                            np.load(config['gain'])['arr_0'],
                            np.load(config['offset'])['arr_0'],
                            np.load(config['var'])['arr_0']]  
          
    def marginal_likelihood(self,idx,adu,nsamples=100):
        paramgt = self.thetagt[idx]
        bounds = [3,3,1,300]
        pbound = bounds[idx]
        param_space = np.linspace(paramgt-pbound,paramgt+pbound,nsamples)
        loglike = np.zeros_like(param_space)
        theta_ = np.zeros_like(self.thetagt)
        theta_ = theta_ + self.thetagt
        for n in range(nsamples):
           theta_[idx] = param_space[n]
           loglike[n] = isologlike3d(theta_,adu,self.cmos_params)
        fig,ax=plt.subplots()
        ax.plot(param_space,loglike,color='red')
        ax.vlines(paramgt,ymin=loglike.min(),ymax=loglike.max(),color='black')
                           
    def test(self):
        self.thetagt = np.array([self.config['x0'],
                                 self.config['y0'],
                                 self.config['z0'],
                                 self.config['N0']])
        iso3d = Iso3D(self.thetagt,self.config)
        theta0 = np.zeros_like(self.thetagt)
        theta0[0] = self.thetagt[0] + np.random.normal(0,2)
        theta0[1] = self.thetagt[1] + np.random.normal(0,2)
        theta0[2] = 0.0
        theta0[3] = self.thetagt[3]
        adu = iso3d.generate(plot=True)
        adu = adu - self.cmos_params[3]
        self.marginal_likelihood(2,adu)
        lr = np.array([1e-4,1e-4,1e-4,0]) #this needs to be changed depending on cps
        opt = MLE3D(theta0,adu,self.config,theta_gt=self.thetagt)
        theta, loglike = opt.optimize(max_iters=1000,lr=lr,plot=True)

