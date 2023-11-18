import numpy as np
import pandas as pd
import ripleyk
import matplotlib.pyplot as plt
from SMLM.utils import *
from SMLM.figures import format_ax
from matplotlib.gridspec import GridSpec
from scipy.stats import multivariate_normal
from ..cluster import FixedCovMixture, removeKNearest

class Figure_3:
    """Clustering Figure"""
    def __init__(self, config,var=1e-4):
        self.config = config
        self.var = var

    def fit_mixture(self,X,Nk=40,plot=True):
        nspot, ndim = X.shape
        loglikes = []; BICs = []; mixtures = []
        for n in range(Nk):
            model = FixedCovMixture(nspot-n, var=self.var, random_state=1)
            loglike, gmixture = model.fit(X)
            nparams = 2*nspot-n
            BIC = nparams*np.log(nspot) - 2*loglike
            loglikes.append(loglike); BICs.append(BIC)
            mixtures.append(gmixture)
            
        nspace = np.arange(0,Nk,1)
        return nspace,BICs,loglikes,mixtures
    
    def plot(self, prefixes, centers, pixel_size=108.3, nbins=500):  
        fig = plt.figure(layout="constrained", figsize=(10,4))

        gs = GridSpec(3, 6, figure=fig)
        ax0 = fig.add_subplot(gs[:,:2])
        ax1 = fig.add_subplot(gs[:,2:4],aspect=1.0)
        ax2 = fig.add_subplot(gs[0,4],aspect=1.0)
        ax3 = fig.add_subplot(gs[1,4],aspect=1.0)
        ax4 = fig.add_subplot(gs[2,4],aspect=1.0)
        ax5 = fig.add_subplot(gs[0,5])
        ax6 = fig.add_subplot(gs[1,5])
        ax7 = fig.add_subplot(gs[2,5])
        ax1.set_xticks([]); ax1.set_yticks([])
        ax2.set_xticks([]); ax2.set_yticks([])
        ax3.set_xticks([]); ax3.set_yticks([])
        ax4.set_xticks([]); ax4.set_yticks([])
        
        spots = pd.read_csv(self.config['analpath'] + prefixes[0] + '.csv')
        H,xedges,yedges = np.histogram2d(spots['x_mle'], spots['y_mle'],bins=nbins)
        x_range = (spots['x_mle'].max()-spots['x_mle'].min())*0.1083
        y_range = (spots['y_mle'].max()-spots['y_mle'].min())*0.1083
        x_bin_size = x_range/nbins
        y_bin_size = y_range/nbins
        print(f'X bin size: {x_bin_size} um')
        print(f'Y bin size: {y_bin_size} um')
        ax0.imshow(H,cmap='gray',vmin=0.0,vmax=5.0,aspect=x_bin_size/y_bin_size)
        ax0.set_xticks([]); ax0.set_yticks([])
        ax0.invert_yaxis()
        ax0.set_title('Density')
        ax1.set_title('Pointillist')
        
        ax = [ax1, ax2, ax3, ax4]
        for n, prefix in enumerate(prefixes):
            spots = pd.read_csv(self.config['analpath'] + prefix + '.csv')
            ax[n].scatter(spots['y_mle'], spots['x_mle'], color='black', s=1)

        colors = ['red', 'blue', 'cyan']
        sample_size = (10,10)
        axs = [ax5,ax6,ax7]
        for n, ax in enumerate(axs):
            print('Ripley ' + prefixes[n+1])
            spots = pd.read_csv(self.config['analpath'] + prefixes[n+1] + '.csv')
            radii = np.linspace(0,5,20)
            nspace,bics,loglikes,mixtures = self.fit_mixture(spots[['x_mle','y_mle']].values)
            nstar = nspace[np.argmin(bics)]; best_mixture = mixtures[np.argmin(bics)]
            spots_correct = removeKNearest(spots[['x_mle','y_mle']].values,nstar)
            x = spots['x_mle']; y = spots['y_mle']
            K = ripleyk.calculate_ripley(list(radii),sample_size,d1=x, d2=y, sample_shape='rectangle', boundary_correct=False, CSR_Normalise=False)
            xcorrect = spots_correct[:,0]; ycorrect = spots_correct[:,1]
            Kcorrect = ripleyk.calculate_ripley(list(radii),sample_size,d1=xcorrect, d2=ycorrect, sample_shape='rectangle', boundary_correct=False, CSR_Normalise=False)
            ax5.plot(pixel_size*radii,K,color=colors[n])
            ax5.plot(pixel_size*radii,Kcorrect,color=colors[n],linestyle='--')
            ax6.plot(nspace,bics,color=colors[n])
            ax7.plot(nspace,loglikes,color=colors[n])
            if n == 0:
                ax.legend()
        ax5.set_xlabel('r (nm)')
        ax5.set_ylabel('K(r)')
        ax5.vlines(250,0,50,color='black',linestyle='--',label=r'$\lambda/2\mathrm{NA}$')
        ax6.set_xlabel(r'$\Delta N$')
        ax6.set_ylabel('BIC')
        ax7.set_xlabel('$\Delta N$')
        ax7.set_ylabel('Model loglikelihood')
        
        

        for i, color in enumerate(colors):
            center = centers[i]
            rect = plt.Rectangle(tuple(reversed(center)),5,5,fill=False, edgecolor=color, lw=2)
            ax1.add_patch(rect)

        for i,axis in enumerate([ax2, ax3, ax4]):
            axis.spines['top'].set_color(colors[i])
            axis.spines['bottom'].set_color(colors[i])
            axis.spines['left'].set_color(colors[i])
            axis.spines['right'].set_color(colors[i])
            axis.tick_params(top=False, bottom=False, left=False, right=False)



        

