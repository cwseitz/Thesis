import numpy as np
import matplotlib.pyplot as plt
#from BaseSMLM.torch.dataset import SMLMDataset
#from BaseSMLM.mains.run.pipes import *
#from BaseSMLM.utils import errors2d, errors3d
#from BaseSMLM.psf.psf2d import crlb2d

class CRB2D:
    def __init__(self,config):
        self.config = config
        self.cmos_params = [config['eta'],config['texp'],
                            np.load(config['gain'])['arr_0'],
                            np.load(config['offset'])['arr_0'],
                            np.load(config['var'])['arr_0']]  
             
    def forward(self,N0space,npix):
        crlb_n0 = np.zeros((len(N0space),4))
        for i,n0 in enumerate(N0space):
            theta0 = np.array([3.5,3.5,0.92,n0])
            crlb_n0[i] = crlb2d(theta0,self.cmos_params)
        return crlb_n0


class Figure_1:
    """Performance of fitting methods on 2D and 3D emitter"""
    def __init__(self,config):
        self.config = config
        
    def plot2d(self,prefixes2d_p,prefixes2d_k):
        """For 2D isotropic PSF"""
        print("Plot 2D")
        x1 = [100,500,1000]; x2 = [1,3,5]
        klog_thresholds = self.config['mle_2d']['klog_thresholds']
        plog_thresholds = self.config['mle_2d']['plog_thresholds']
        k_lrs = self.config['mle_2d']['k_lrs']
        p_lrs = self.config['mle_2d']['p_lrs']
        fig, ax = plt.subplots(2,2,figsize=(6,6))
        datapath2d = self.config['mle_2d']['datapath']
        datasets2d_p = [SMLMDataset(datapath2d+prefix,prefix) for prefix in prefixes2d_p]
        datasets2d_k = [SMLMDataset(datapath2d+prefix,prefix) for prefix in prefixes2d_k]

        self.add_2d_lateral_rmse_and_jaccard(x1,ax[0,0],ax[0,1],datasets2d_p,plog_thresholds,p_lrs,logx=True)
        self.add_2d_lateral_rmse_and_jaccard(x2,ax[1,0],ax[1,1],datasets2d_k,klog_thresholds,k_lrs)
        self.add_2d_crlb(ax[0,0],[100,500,1000],self.config['mle_2d'])
                
        ax[0,0].set_xlabel('Photons'); ax[0,1].set_xlabel('Photons')
        ax[0,0].set_ylabel('Lateral RMSE (nm)'); ax[0,1].set_ylabel('Jaccard Index')
        ax[1,0].set_xlabel(r'$K(\lambda/2\mathrm{NA})$'); ax[1,1].set_xlabel(r'$K(\lambda/2\mathrm{NA})$')
        ax[1,0].set_ylabel('Lateral RMSE (nm)'); ax[1,1].set_ylabel('Jaccard Index')
        ax[0,0].set_title(r'$K(\lambda/2\mathrm{NA})=1$')
        ax[0,1].set_title(r'$K(\lambda/2\mathrm{NA})=1$')
        ax[1,0].set_title(r'$N_{0}=1000$')
        ax[1,1].set_title(r'$N_{0}=1000$')
        ax[0,1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        ax[1,1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.tight_layout()
        
    def plot3d(self,prefixes3d_p,prefixes3d_k):
        """For 3D astigmatism PSF"""
        print("Plot 3D")
        x1 = [1000,10000,100000]; x2 = [1,3,5]
        klog_thresholds = self.config['mle_3d']['klog_thresholds']
        plog_thresholds = self.config['mle_3d']['plog_thresholds']
        k_lrs = self.config['mle_3d']['k_lrs']
        p_lrs = self.config['mle_3d']['p_lrs']
        pixel_size = self.config['mle_3d']['pixel_size_lateral']
        fig, ax = plt.subplots(2,3,figsize=(8,5))
        datapath3d = self.config['cnn_3d']['datapath']
        datasets3d_p = [SMLMDataset(datapath3d+prefix,prefix) for prefix in prefixes3d_p]
        datasets3d_k = [SMLMDataset(datapath3d+prefix,prefix) for prefix in prefixes3d_k]
        self.add_3d_rmse_and_jaccard(x1,datasets3d_p,plog_thresholds,p_lrs,ax[0,0],ax[0,1],ax[0,2],pixel_size=pixel_size,logx=True,legend=True)
        self.add_3d_rmse_and_jaccard(x2,datasets3d_k,klog_thresholds,k_lrs,ax[1,0],ax[1,1],ax[1,2],pixel_size=pixel_size)
        ax[0,0].set_xlabel('cps'); ax[0,1].set_xlabel('cps'); ax[0,2].set_xlabel('cps')
        ax[0,0].set_ylabel('Lateral RMSE (nm)'); ax[0,1].set_ylabel('Axial RMSE (nm)'); ax[0,2].set_ylabel('Jaccard Index')
        ax[1,0].set_xlabel(r'$K(\lambda/2\mathrm{NA})$'); ax[1,1].set_xlabel(r'$K(\lambda/2\mathrm{NA})$'); 
        ax[1,2].set_xlabel(r'$K(\lambda/2\mathrm{NA})$')
        ax[1,0].set_ylabel('Lateral RMSE (nm)'); ax[1,1].set_ylabel('Axial RMSE (nm)'); ax[1,2].set_ylabel('Jaccard Index')
        plt.tight_layout()
        plt.show()

    def add_2d_crlb(self,ax,N0space,config,pixel_size=108.3):
        crb = CRB2D(config)
        crbn0 = crb.forward(N0space,7)
        ax.semilogx(N0space,pixel_size*crbn0[:,0],color='gray',label=r'$\sigma_{CRLB}$')
        

    def add_2d_lateral_rmse_and_jaccard(self,x,ax1,ax2,datasets2d,thresholds,lrs,pixel_size=108.3,logx=False):
        """Add the lateral RMSE and jaccard index for 
           MLE and CNN as a function of photon count in the isolated case (K=1)"""
        kerr_mle = []; kjac_mle = []
        kerr_cnn = []; kjac_cnn = []
        
        for n,dataset in enumerate(datasets2d):
            pipe_cnn = PipelineCNN2D(self.config['cnn_2d'],dataset)
            spots_cnn = pipe_cnn.localize()
            xerr, yerr, jacc = errors2d(spots_cnn,dataset.theta)
            rmse = pixel_size*np.sqrt(np.var(yerr))
            kerr_cnn.append(rmse); kjac_cnn.append(jacc)

            mle2d_config = self.config['mle_2d'].copy()
            mle2d_config['thresh_log'] = thresholds[n]
            
            dataset.stack = np.squeeze(dataset.stack)
            pipe_mle = PipelineMLE2D(mle2d_config,dataset)
            spots_mle = pipe_mle.localize(plot=False,lr=lrs[n])
            spots_mle.drop(columns=(['x','y']),inplace=True)
            spots_mle = spots_mle.rename(columns={"x_mle": "x", "y_mle": "y"})
            xerr, yerr, jacc = errors2d(spots_mle,dataset.theta)
            rmse = pixel_size*np.sqrt(np.var(xerr))
            kerr_mle.append(rmse); kjac_mle.append(jacc)
            
        if logx:
            ax1.semilogx(x,kerr_mle,color='red',marker='x')
            ax1.semilogx(x,kerr_cnn,color='blue',marker='x')
            ax2.semilogx(x,kjac_mle,color='red',marker='x',label='MLE')
            ax2.semilogx(x,kjac_cnn,color='blue',marker='x',label='CNN')
        else:
            ax1.plot(x,kerr_mle,color='red',marker='x')
            ax1.plot(x,kerr_cnn,color='blue',marker='x')
            ax2.plot(x,kjac_mle,color='red',marker='x',label='MLE')
            ax2.plot(x,kjac_cnn,color='blue',marker='x',label='CNN')
                  
    def add_3d_rmse_and_jaccard(self,x,datasets3d,thresholds,lrs,ax1,ax2,ax3,logx=False,pixel_size=108.3,legend=False):
        """Add the lateral RMSE for MLE and CNN as a function of photon count"""
        kerr_mle_lateral = []; kerr_mle_axial = []; kjac_mle = []
        kerr_cnn_lateral = []; kerr_cnn_axial = []; kjac_cnn = []


        for n,dataset in enumerate(datasets3d):
              
            dataset.stack = np.squeeze(dataset.stack)
            
            mle3d_config = self.config['mle_3d'].copy()
            mle3d_config['thresh_log'] = thresholds[n]
            
            pipe_mle = PipelineMLE3D(mle3d_config,dataset)
            spots_mle = pipe_mle.localize(plot=False,lr=lrs[n],patchw=11)
            
            spots_mle.drop(columns=(['x','y']),inplace=True)
            spots_mle = spots_mle.rename(columns={"x_mle": "x", 
                                                  "y_mle": "y",
                                                  "z_mle": "z"})
            xerr, yerr, zerr, jacc = errors3d(spots_mle,dataset.theta)
            rmse_lateral = pixel_size*np.sqrt(np.var(xerr))
            rmse_axial = 1000*np.sqrt(np.var(zerr))
            kerr_mle_lateral.append(rmse_lateral)
            kerr_mle_axial.append(rmse_axial)
            kjac_mle.append(jacc) 

            """
            dataset.stack = np.expand_dims(dataset.stack,1)
            pipe_cnn = PipelineCNN3D(self.config['cnn_3d'],dataset)
            spots_cnn = pipe_cnn.localize()
            xerr, yerr, zerr, jacc = errors3d(spots_cnn,dataset.theta)
            rmse_lateral = pixel_size*np.sqrt(np.var(xerr))
            rmse_axial = 1000*np.sqrt(np.var(zerr)) #convert to nm
            kerr_cnn_lateral.append(rmse_lateral)
            kerr_cnn_axial.append(rmse_axial)
            kjac_cnn.append(jacc)
            """
        
        if logx:
            #ax1.semilogx(x,kerr_cnn_lateral,color='red',marker='x')
            ax1.semilogx(x,kerr_mle_lateral,color='blue',marker='x')
        else:
            #ax1.plot(x,kerr_cnn_lateral,color='red',marker='x')
            ax1.plot(x,kerr_mle_lateral,color='blue',marker='x')
        
        if logx:
            #ax2.semilogx(x,kerr_cnn_axial,color='red',marker='x')
            ax2.semilogx(x,kerr_mle_axial,color='blue',marker='x')
        else:
            #ax2.plot(x,kerr_cnn_axial,color='red',marker='x')
            ax2.plot(x,kerr_mle_axial,color='blue',marker='x')
        
        if legend:
            if logx:
                #ax3.semilogx(x,kjac_cnn,color='red',label='CNN',marker='x')
                ax3.semilogx(x,kjac_mle,color='blue',label='MLE',marker='x')
                ax3.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            else:
                #ax3.plot(x,kjac_cnn,color='red',label='CNN',marker='x')
                ax3.plot(x,kjac_mle,color='blue',label='MLE',marker='x')
                ax3.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        if not legend:
            if logx:
                #ax3.semilogx(x,kjac_cnn,color='red',label='CNN\ncps=1000',marker='x')
                ax3.semilogx(x,kjac_mle,color='blue',label='MLE\ncps=1000',marker='x')
                ax3.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            else:
                #ax3.plot(x,kjac_cnn,color='red',label='CNN\ncps=1000',marker='x')
                ax3.plot(x,kjac_mle,color='blue',label='MLE\ncps=1000',marker='x')
                ax3.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            
        plt.tight_layout()




