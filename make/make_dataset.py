from dataset import *
from BaseSMLM.utils import RLDeconvolver

nx = ny = 20
radius = 1.5
nspots = 3
args = [radius,nspots]
kwargs = {'N0':1000,'sigma_ring':0.1}
generator = GaussianRing2D(nx,ny)
dataset = Dataset(1000)
X,Y,Z,S = dataset.make_dataset(generator,args,kwargs,show=False,
                               interpolate=False,upsample=4)
savepath = '/home/cwseitz/Desktop/Torch/Diffusion/'
prefix = '240307_Diffusion'
imsave(savepath+prefix+'_lr.tif',X)
imsave(savepath+prefix+'_sr.tif',Y)
imsave(savepath+prefix+'_hr.tif',Z)

rl = RLDeconvolver()
rl.deconvolve(X[0],plot=True)
