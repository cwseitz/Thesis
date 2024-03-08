import numpy as np
import json
import matplotlib.pyplot as plt
from BaseSMLM.generators import *
from BaseSMLM.utils import BasicKDE
from skimage.io import imsave
from skimage import exposure
from skimage.filters import gaussian
from diffusion.interpolate import G2
from scipy.ndimage import zoom
from skimage.exposure import rescale_intensity

class Dataset:
    """Unified dataset object"""
    def __init__(self,ngenerate):
        self.ngenerate = ngenerate
        self.X_type = np.int16
        self.Z_type = np.float32
    def show(self,X,Y,Z):
        fig,ax=plt.subplots(1,3)
        nx,ny = Z.shape
        RGB = np.zeros((nx,ny,3))
        RGB[:,:,0] = Z/Z.max()
        RGB[:,:,2] = Y/Y.max()
        ax[0].imshow(X,cmap='gray'); ax[0].set_title('LR')
        ax[1].imshow(Y,cmap='gray'); ax[1].set_title('SPLINE')
        ax[2].imshow(RGB); ax[2].set_title('KDE/SPLINE')
        plt.tight_layout()
        plt.show()    
    def make_dataset(self,generator,args,kwargs,upsample=8,interpolate=False,show=False):
        pad = upsample // 2
        Xs = []; Ys = []; Zs = []; Ss = []
        for n in range(self.ngenerate):
            print(f'Generating sample {n}')
            G = generator.forward(*args,**kwargs)
            theta = G[2][:2,:].T; S = G[1]
            X = rescale_intensity(G[0],out_range=self.X_type)
            if interpolate:
                X = rescale_intensity(G2(G[0]),out_range=self.X_type)
                X = gaussian(X,sigma=1.0,preserve_range=True)
                theta = 2*theta
            nx,ny = X.shape
            Y = zoom(X,(upsample,upsample),order=3)
            Z = BasicKDE(theta).forward(nx,upsample=upsample,sigma=3.0)
            Z = np.pad(Z,((pad,0),(pad,0)))
            Z = Z[:-pad,:-pad]
            Z = rescale_intensity(Z,out_range=self.Z_type)
            Xs.append(X); Ys.append(Y); Zs.append(Z); Ss.append(S)
            if show:
                self.show(X,Y,Z)
        return (np.array(Xs),np.array(Ys),np.array(Zs),np.array(Ss))
 

