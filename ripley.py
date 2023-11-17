import numpy as np
import pandas as pd
import ripleyk
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from ..cluster import FixedCovMixture, removeKNearest


def fit_mixture(X,Nk=40,var=1e-4):
    nspot, ndim = X.shape
    loglikes = []; BICs = []; mixtures = []
    for n in range(Nk):
        print(f'Loglikelihood K={n}')
        model = FixedCovMixture(nspot-n, var=var, random_state=1)
        loglike, gmixture = model.fit(X)
        nparams = 2*nspot-n
        BIC = nparams*np.log(nspot) - 2*loglike
        loglikes.append(loglike); BICs.append(BIC)
        mixtures.append(gmixture)
        
    nspace = np.arange(0,Nk,1)
    return nspace,BICs,loglikes,mixtures

def ripley(spots,radii=None,sample_size=(1,1),correct=False,plot_gmm=False): 
    if radii is None:
        radii = np.linspace(0,5,20)
    if correct:
        nspace,bics,loglikes,mixtures = fit_mixture(spots[['x_mle','y_mle']].values)
        if plot_gmm:
            fig,ax=plt.subplots(1,2)
            ax[0].plot(nspace,bics)
            ax[1].plot(nspace,loglikes)
            plt.show()
        nstar = nspace[np.argmin(bics)]; best_mixture = mixtures[np.argmin(bics)]
        spots_correct = removeKNearest(spots[['x_mle','y_mle']].values,nstar)  
              
    x = spots['x_mle']; y = spots['y_mle']
    K = ripleyk.calculate_ripley(list(radii),sample_size,d1=x, d2=y, sample_shape='rectangle', boundary_correct=False, CSR_Normalise=False)

    return K,radii


        

