import numpy as np
import umap
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
        opt = MLE2DMix(theta0,adu,self.config,theta_gt=thetagt)
        theta, thetat, loglike, conv = opt.optimize(max_iters=self.config['max_iters'],lr=lr,plot_fit=True)
        if plot:
            self.show(adu,loglike,thetat,theta,theta0,thetagt)
            
    def plot_umap_with_time_coloring(self,data,time,loglike):
        fig,ax=plt.subplots(1,2)
        reducer = umap.UMAP(n_neighbors=15, n_components=2, random_state=42)
        embedding = reducer.fit_transform(data)
        scatter1 = ax[0].scatter(embedding[:,0], embedding[:,1], c=time, cmap='coolwarm', s=10, alpha=0.8)
        scatter2 = ax[1].scatter(embedding[:,0], embedding[:,1], c=loglike, cmap='plasma', s=10, alpha=0.8)
        cbar1 = plt.colorbar(scatter1,ax=ax[0])
        cbar2 = plt.colorbar(scatter2,ax=ax[1])
        ax[0].set_xlabel('UMAP Dimension 1')
        ax[0].set_ylabel('UMAP Dimension 2')
        ax[1].set_xlabel('UMAP Dimension 1')
        ax[1].set_ylabel('UMAP Dimension 2')
                        
    def show(self,adu,loglike,thetat,theta,theta0,thetagt):
        fig,ax=plt.subplots()
        ax.imshow(adu,cmap='gray')
        ax.scatter(theta[1,:],theta[0,:],marker='x',s=5,color='red')
        ax.scatter(theta0[1,:],theta0[0,:],marker='x',s=5,color='blue')
        ax.scatter(thetagt[1,:],thetagt[0,:],marker='x',s=5,color='green')
        ntheta,nspots = theta.shape
        thetat_cut = thetat[:,:2,:]
        nt,_,_ = thetat_cut.shape
        thetat_cut = thetat_cut.reshape((nt,2*nspots))
        time = np.arange(0,thetat.shape[0],1)
        
        self.plot_umap_with_time_coloring(thetat_cut,time,loglike)
        plt.show()
        fig,ax=plt.subplots(nspots,ntheta)
        for n in range(ntheta):
            for m in range(nspots):
               ax[m,n].plot(thetat[:,n,m],color='black')
        plt.tight_layout()
        plt.show()

