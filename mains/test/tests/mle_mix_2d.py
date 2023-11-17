import numpy as np
import matplotlib.pyplot as plt
from BaseSMLM.generators import Mix2D
from BaseSMLM.psf.psf2d import *

class MLE2DMix_Test:
    """Test a single instance of maximum likelihood estimation"""
    def __init__(self,setup_params):
        self.setup_params = setup_params
        self.cmos_params = [setup_params['eta'],setup_params['texp'],
                            np.load(setup_params['gain'])['arr_0'],
                            np.load(setup_params['offset'])['arr_0'],
                            np.load(setup_params['var'])['arr_0']]  

    def marginal_likelihood(self,idx,adu,nsamples=100):
        paramgt = self.thetagt[idx]
        bounds = [3,3,0.5,5000]
        pbound = bounds[idx]
        param_space = np.linspace(paramgt-1000,paramgt+pbound,nsamples)
        loglike = np.zeros_like(param_space)
        theta_ = np.zeros_like(self.thetagt)
        theta_ = theta_ + self.thetagt
        for n in range(nsamples):
           theta_[idx] = param_space[n]
           loglike[n] = isologlike2d(theta_,adu,self.cmos_params)
        fig,ax=plt.subplots()
        ax.plot(param_space,loglike,color='red')
        ax.vlines(paramgt,ymin=loglike.min(),ymax=loglike.max(),color='black')
    def get_errors(self,theta,adu):
        hess = hessiso_auto2d(theta,adu,self.cmos_params)
        errors = np.sqrt(np.diag(inv(hess)))
        return errors
    def test(self):
        nx,ny = self.cmos_params[2].shape
        self.thetagt = np.array([self.setup_params['x0'],
                                 self.setup_params['y0'],
                                 self.setup_params['sigma'],
                                 self.setup_params['N0']])
        iso2d = Iso2D(self.thetagt,self.setup_params)
        adu = iso2d.generate(plot=True)
        adu = adu - self.cmos_params[5]
        adu = np.clip(adu,0,None)
        lr = np.array([0.001,0.001,0.0,100.0]) #hyperpar
        theta0 = np.zeros_like(self.thetagt)
        theta0 += self.thetagt
        theta0[0] += np.random.normal(0,1)
        theta0[1] += np.random.normal(0,1)
        theta0[3] += 100
        self.marginal_likelihood(3,adu)
        opt = MLEOptimizer2DGrad(theta0,adu,self.setup_params,theta_gt=self.thetagt)
        theta, loglike = opt.optimize(iters=100,lr=lr,plot=True,grid_search=True)
        error_mle = self.get_errors(theta,adu)
        print(error_mle)
        
