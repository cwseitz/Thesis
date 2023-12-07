import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from BaseSMLM.utils import KDE, FRC

class Figure_4:
    """Fourier Ring Correlation"""
    def __init__(self,config,prefix):
        self.config = config
        self.spots = pd.read_csv(self.config['analpath'] + prefix + '.csv')

    def plot(self,niters=10):
        spots = self.spots
        fracs = [0.05,0.1,0.2,0.4]
        colors = ['royalblue','gray','red','darkorchid']
        plt.rcParams['font.family'] = 'Sans-serif'

        frc_res_ = np.zeros((niters,len(fracs)))
        frc_curve_ = np.zeros((niters,len(fracs),2100)) #depends on KDE pixel size

        nlocal = len(spots)
        for m in range(niters):
            x_range = (0,400); y_range = (0,400)
            spots = spots.loc[(spots['x_mle'] > x_range[0]) & (spots['x_mle'] < x_range[1])]
            spots = spots.loc[(spots['y_mle'] > y_range[0]) & (spots['y_mle'] < y_range[1])]
            for n,frac in enumerate(fracs):
                nsamples = int(round(frac*nlocal))
                print(f'Iteration {m}, Fraction: {frac}')
                frc = FRC(spots)
                freq, thres, frc_curve, res = frc.compute_frc(x_range,y_range,nsamples=nsamples,sigma=2.0,scale=100.0,plot_kde=False,plot_fft=False)
                frc_res_[m,n] = res
                frc_curve_[m,n] = frc_curve
               
        avg_frc = np.mean(frc_curve_,axis=0)
        std_frc = np.std(frc_curve_,axis=0)
        
        fig,ax=plt.subplots(2,1,figsize=(3,6))
        for n in range(len(fracs)):
            ax[0].errorbar(freq,avg_frc[n],yerr=std_frc[n],color=colors[n],label=fracs[n])
        ax[0].plot(freq,thres,color='black',linestyle='--')
        ax[0].set_xlabel(r'Spatial frequency q ($\mathrm{um}^{-1}$)')
        ax[0].set_ylabel('FRC')
        ax[0].legend()


        avg = np.mean(frc_res_,axis=0)
        std = np.std(frc_res_,axis=0)
        ax[1].errorbar(fracs,avg,yerr=std,color='royalblue')
        ax[1].set_xlabel('Fraction of localizations')
        ax[1].set_ylabel(r'$R\;\mathrm{(um)}$',color='royalblue')
        ax2 = ax[1].twinx()
        ax2.plot(fracs,1/avg,color='red',marker='o')
        ax2.set_ylabel(r'$q_{R}$ ($\mathrm{um}^{-1}$)',color='red')
        plt.tight_layout()
        plt.show()
