import numpy as np
import matplotlib.pyplot as plt
from BaseSMLM.generators import Disc2D
from BaseSMLM.psf.psf2d import *

class MLE2D_BFGS_Test:
    def __init__(self,config):
        self.config = config
        self.cmos_params = [config['eta'],config['texp'],
                            np.load(config['gain'])['arr_0'],
                            np.load(config['offset'])['arr_0'],
                            np.load(config['var'])['arr_0']]  
    def test(self,niters=1):
        gain,offset,var = self.cmos_params[2:]
        nx,ny = offset.shape
        for n in range(niters):
            disc2d = Disc2D(nx,ny)
            adu,spikes,thetagt = disc2d.forward(3.0,1,show=False,N0=1000.0)
            adu = adu - offset
            adu = np.clip(adu,0,None)
            theta0 = np.zeros((3,))
            theta0 += np.delete(thetagt,2)
            theta0[0] += np.random.normal(0,1)
            theta0[1] += np.random.normal(0,1)
            theta0[2] += np.random.normal(0,10)
            opt = MLE2D_BFGS(theta0,adu,self.config,theta_gt=thetagt)
            theta,loglike,conv,err = opt.optimize(max_iters=100,plot_fit=False)



       
