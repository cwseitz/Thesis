from dataset import *
from BaseSMLM.utils import RLDeconvolver

nx = ny = 20
radius = 0.92
nspots = None
args = [radius,nspots]
kwargs = {'N0':1000,'sigma_ring':0.1,'rand_center':False}
generator = GaussianRing2D(nx,ny)
dataset = Dataset(1000)
X,Y,Z,S = dataset.make_dataset(generator,args,kwargs,show=False,
                               interpolate=False,upsample=4,sigma_kde=1.5,sigma_gauss=0.5)
savepath = '/home/cwseitz/Desktop/Torch/Diffusion/'
prefix = '240317_Diffusion'
imsave(savepath+prefix+'_lr.tif',X)
imsave(savepath+prefix+'_sr.tif',Y)
imsave(savepath+prefix+'_hr.tif',Z)

rl = RLDeconvolver()
rl.deconvolve(X[0],plot=True)
