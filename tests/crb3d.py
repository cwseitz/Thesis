import numpy as np
import matplotlib.pyplot as plt
from SMLM.generators import Iso3D
from SMLM.psf import *
from scipy.stats import multivariate_normal

class CRB3D_Test1:
    """Fixed axial position and variable photon count (SNR)"""
    def __init__(self,config,error_samples=500,max_iters=1000):
        self.config = config
        self.max_iters = max_iters
        self.cmos_params = [config['eta'],config['texp'],
                            np.load(config['gain'])['arr_0'],
                            np.load(config['offset'])['arr_0'],
                            np.load(config['var'])['arr_0']]  
        self.thetagt = np.array([self.config['x0'],
                                 self.config['y0'],
                                 self.config['z0'],
                                 self.config['N0']])
        self.thetagt[2] = 0
        self.error_samples = error_samples
        
    def plot(self,ax1,ax2,nn=5):
        pixel_size = self.config['pixel_size_lateral']
        N0space = np.linspace(100,1000,nn)
        rmse_mle = self.rmse_mle_batch(N0space)
        crlb_n0 = self.crlb(N0space,self.thetagt)
        ax1.semilogx(N0space,pixel_size*crlb_n0[:,0],color='cornflowerblue',marker='o',label='CRLB')
        ax1.semilogx(N0space,pixel_size*rmse_mle[:,0],color='cornflowerblue',marker='x')
        ax2.semilogx(N0space,rmse_mle[:,2],color='purple',marker='x',label='z')
        ax1.set_xlabel(r'$N_{0}$')
        ax1.set_ylabel('Lateral error (nm)',color='cornflowerblue')
        ax2.set_ylabel('Axial error (nm)',color='purple')
        ax1.legend()
        #ax2.legend()
        ax1.set_title(r'$z_{0}$ = 0')
        plt.tight_layout()
        
    def rmse_mle3d(self,n0):
        err = np.zeros((self.error_samples,4))
        theta = np.zeros_like(self.thetagt)
        theta += self.thetagt
        theta[3] = n0
        for n in range(self.error_samples):
            print(f'Error sample {n}')
            iso3d = Iso3D(theta,self.config)
            adu = iso3d.generate(plot=False)
            adu = np.clip(adu-self.cmos_params[3],0,None)
            theta0 = np.zeros_like(self.thetagt)
            theta0 += theta
            theta0[0] += np.random.normal(0,1)
            theta0[1] += np.random.normal(0,1)
            theta0[2] += np.random.normal(0,100)
            opt = MLE3D(theta0,adu,self.config,theta_gt=theta)
            theta_est,loglike = opt.optimize(max_iters=self.max_iters,lr=self.config['lr'],plot=False)
            err[n,:] = theta_est - self.thetagt
            del iso3d
        return np.sqrt(np.mean(err**2,axis=0))
               
    def rmse_mle_batch(self,N0space):
        errs = np.zeros((len(N0space),4))
        for i,n0 in enumerate(N0space):
            errs[i] = self.rmse_mle3d(n0)
        return errs
 
    def rmse_sgld_batch(self,N0space):
        errs = np.zeros((len(N0space),5))
        for i,n0 in enumerate(N0space):
            errs[i] = self.rmse_sgld3d(n0)
        return errs
        
    def crlb(self,N0space,theta0,nn=5):
        """Assume z0 = 0 and you can use the 2D CRLB"""
        crlb_n0 = np.zeros((nn,5))
        for i,n0 in enumerate(N0space):
            theta0[4] = n0
            crlb_n0[i] = crlb3d(theta0,self.cmos_params)
        return crlb_n0

class CRB3D_Test2:
    """Fixed photon count (SNR) and variable axial position"""
    def __init__(self,config,error_samples=500,max_iters=1000):
        self.config = config
        self.max_iters = max_iters
        self.cmos_params = [config['eta'],config['texp'],
                            np.load(config['gain'])['arr_0'],
                            np.load(config['offset'])['arr_0'],
                            np.load(config['var'])['arr_0']]  
        self.thetagt = np.array([self.config['x0'],
                                 self.config['y0'],
                                 self.config['z0'],
                                 self.config['N0']]) 
        self.error_samples = error_samples
        
    def plot(self,ax1,ax2):
        theta0 = np.zeros_like(self.thetagt)
        theta0 += self.thetagt
        z0space = np.linspace(-0.4,0.4,10)
        rmse = self.rmse_mle_batch(z0space)
        pixel_size = self.config['pixel_size_lateral']
        ax1.plot(1e3*z0space,pixel_size*rmse[:,0],color='cornflowerblue',marker='x',label='x')
        ax1.plot(1e3*z0space,pixel_size*rmse[:,1],color='blue',marker='x',label='y')
        ax2.plot(1e3*z0space,1000*rmse[:,2],color='darkorchid',marker='x',label='z')
        ax1.set_xlabel('z (nm)')
        ax1.set_ylabel('Lateral RMSE (nm)',color='black')
        ax2.set_xlabel('z (nm)')
        ax2.set_ylabel('Axial RMSE (nm)',color='black')
        ax1.legend()
        ax2.legend()
        ax1.set_title(r'$N_{0}=10^{4}$ photons')
        plt.tight_layout()
        
    def rmse_mle3d(self,z0):
        err = np.zeros((self.error_samples,4))
        theta = np.zeros_like(self.thetagt)
        theta = theta + self.thetagt
        theta[2] = z0
        for n in range(self.error_samples):
            print(f'Error sample {n}')
            iso3d = Iso3D(theta,self.config)
            adu = iso3d.generate(plot=False)
            adu = np.clip(adu-self.cmos_params[3],0,None)
            theta0 = np.zeros_like(self.thetagt)
            theta0 = theta0 + theta
            theta0[0] += np.random.normal(0,1)
            theta0[1] += np.random.normal(0,1)
            theta0[2] = 0
            opt = MLE3D(theta0,adu,self.config,theta_gt=theta)
            theta_est,loglike = opt.optimize(max_iters=self.max_iters,lr=self.config['lr'],plot=False)
            err[n,:] = theta_est - theta
            del iso3d
        return np.sqrt(np.var(err,axis=0))
           
    def rmse_mle_batch(self,z0space):
        errs = np.zeros((len(z0space),4))
        for i,z0 in enumerate(z0space):
            errs[i] = self.rmse_mle3d(z0)
        return errs
   
class CRB3D_Test3:
    """Fixed SNR, fixed axial position, variable zmin, variable A/B"""
    def __init__(self,config):
        self.config = config
        self.cmos_params = [config['nx'],config['ny'],
                           config['eta'],config['texp'],
                            np.load(config['gain'])['arr_0'],
                            np.load(config['offset'])['arr_0'],
                            np.load(config['var'])['arr_0']]  
        self.thetagt = np.array([self.config['x0'],
                                 self.config['y0'],
                                 self.config['z0'],
                                 self.config['N0']]) 
        
    def plot(self,ax):
        zminspace = np.linspace(10,1000,100)
        abspace = np.linspace(1e-7,1e-5,100)
        theta0 = np.zeros_like(self.thetagt)
        theta0 += self.thetagt
        crlb_map = self.crlb(zminspace,abspace,theta0)
        ax[0].imshow(crlb_map[:,:,0],cmap='jet')
        ax[0].set_title('x')
        ax[0].set_xlabel(r'$z_{min}$')
        ax[0].set_ylabel(r'$\alpha$')
        ax[1].imshow(crlb_map[:,:,1],cmap='jet')
        ax[1].set_title('y')
        ax[1].set_xlabel(r'$z_{min}$')
        ax[1].set_ylabel(r'$\alpha$')
        ax[2].imshow(crlb_map[:,:,2],cmap='jet')
        ax[2].set_title('z')
        ax[2].set_xlabel(r'$z_{min}$')
        ax[2].set_ylabel(r'$\alpha$')
        plt.tight_layout()
        plt.show()
        
    def crlb(self,zminspace,abspace,theta0):
        crlb_map = np.zeros((100,100,5))
        for i,zmin in enumerate(zminspace):
            for j,ab in enumerate(abspace):
                print(f'CRLB Map [{i},{j}]')
                crlb_map[i,j] = crlb3d(theta0,self.cmos_params)
        return crlb_map     
