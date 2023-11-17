from pycromanager import Dataset
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import tifffile

class Tiler4c:
    def __init__(self,datapath,analpath,prefix,overlap=204):
        self.overlap = overlap
        self.datapath = datapath
        self.analpath = analpath
        self.prefix = prefix
    def blockshaped(self,arr,nrows,ncols):
        h, w = arr.shape
        assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
        assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
        return (arr.reshape(h//nrows, nrows, -1, ncols)
                   .swapaxes(1,2)
                   .reshape(-1, nrows, ncols))
    def tile(self):
        dataset = Dataset(self.datapath+self.prefix)
        X = dataset.as_array(stitched=False,axes=['z','channel','row','column'])
        nz,nc,nt,_,nx,ny = X.shape
        print('Tiling Channel 0\n')
        ch0 = np.asarray(X[:,0,:,:,:-self.overlap,:-self.overlap])
        ch0 = np.max(ch0,axis=0) #max intensity projection
        ch0 = ch0.swapaxes(1,2)
        ch0 = ch0.reshape((nt*(nx-self.overlap),nt*(ny-self.overlap)))
        tifffile.imwrite(self.analpath+self.prefix+'/'+self.prefix+'_mxtiled_ch0.tif',ch0)
        ch0_blocks = self.blockshaped(ch0,1844,1844)
        path = self.analpath+self.prefix+'/'+self.prefix+'_mxtiled_stack_ch0.tif'
        tifffile.imwrite(path,ch0_blocks)
        del ch0, ch0_blocks
        print('Tiling Channel 1\n')
        ch1 = np.asarray(X[:,1,:,:,:-self.overlap,:-self.overlap])
        ch1 = np.max(ch1,axis=0) #max intensity projection
        ch1 = ch1.swapaxes(1,2)
        ch1 = ch1.reshape((nt*(nx-self.overlap),nt*(ny-self.overlap)))
        tifffile.imwrite(self.analpath+self.prefix+'/'+self.prefix+'_mxtiled_ch1.tif',ch1)
        ch1_blocks = self.blockshaped(ch1,1844,1844)
        path = self.analpath+self.prefix+'/'+self.prefix+'_mxtiled_stack_ch1.tif'
        tifffile.imwrite(path,ch1_blocks)
        del ch1, ch1_blocks
        print('Tiling Channel 2\n')
        ch2 = np.asarray(X[:,2,:,:,:-self.overlap,:-self.overlap])
        ch2 = np.max(ch2,axis=0) #max intensity projection
        ch2 = ch2.swapaxes(1,2)
        ch2 = ch2.reshape((nt*(nx-self.overlap),nt*(ny-self.overlap)))
        tifffile.imwrite(self.analpath+self.prefix+'/'+self.prefix+'_mxtiled_ch2.tif',ch2)
        del ch2
        print('Tiling Channel 3\n')
        ch3 = np.asarray(X[:,3,:,:,:-self.overlap,:-self.overlap])
        ch3 = np.max(ch3,axis=0) #max intensity projection
        ch3 = ch3.swapaxes(1,2)
        ch3 = ch3.reshape((nt*(nx-self.overlap),nt*(ny-self.overlap)))
        tifffile.imwrite(self.analpath+self.prefix+'/'+self.prefix+'_mxtiled_ch3.tif',ch3)
        del ch3
        
class Tiler3c:
    def __init__(self,datapath,analpath,prefix,overlap=204):
        self.overlap = overlap
        self.datapath = datapath
        self.analpath = analpath
        self.prefix = prefix
    def blockshaped(self,arr,nrows,ncols):
        h, w = arr.shape
        assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
        assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
        return (arr.reshape(h//nrows, nrows, -1, ncols)
                   .swapaxes(1,2)
                   .reshape(-1, nrows, ncols))
    def tile(self):
        dataset = Dataset(self.datapath+self.prefix)
        X = dataset.as_array(stitched=False,axes=['z','channel','row','column'])
        nz,nc,nt,_,nx,ny = X.shape
        print('Tiling Channel 0\n')
        ch0 = np.asarray(X[:,0,:,:,:-self.overlap,:-self.overlap])
        ch0 = np.max(ch0,axis=0) #max intensity projection
        ch0 = ch0.swapaxes(1,2)
        ch0 = ch0.reshape((nt*(nx-self.overlap),nt*(ny-self.overlap)))
        tifffile.imwrite(self.analpath+self.prefix+'/'+self.prefix+'_mxtiled_ch0.tif',ch0)
        ch0_blocks = self.blockshaped(ch0,1844,1844)
        path = self.analpath+self.prefix+'/'+self.prefix+'_mxtiled_stack_ch0.tif'
        tifffile.imwrite(path,ch0_blocks)
        del ch0, ch0_blocks
        print('Tiling Channel 1\n')
        ch1 = np.asarray(X[:,1,:,:,:-self.overlap,:-self.overlap])
        ch1 = np.max(ch1,axis=0) #max intensity projection
        ch1 = ch1.swapaxes(1,2)
        ch1 = ch1.reshape((nt*(nx-self.overlap),nt*(ny-self.overlap)))
        tifffile.imwrite(self.analpath+self.prefix+'/'+self.prefix+'_mxtiled_ch1.tif',ch1)
        ch1_blocks = self.blockshaped(ch1,1844,1844)
        path = self.analpath+self.prefix+'/'+self.prefix+'_mxtiled_stack_ch1.tif'
        tifffile.imwrite(path,ch1_blocks)
        del ch1, ch1_blocks
        print('Tiling Channel 2\n')
        ch2 = np.asarray(X[:,2,:,:,:-self.overlap,:-self.overlap])
        ch2 = np.max(ch2,axis=0) #max intensity projection
        ch2 = ch2.swapaxes(1,2)
        ch2 = ch2.reshape((nt*(nx-self.overlap),nt*(ny-self.overlap)))
        tifffile.imwrite(self.analpath+self.prefix+'/'+self.prefix+'_mxtiled_ch2.tif',ch2)
        del ch2

class Tiler5c:
    def __init__(self,datapath,analpath,prefix,overlap=204):
        self.overlap = overlap
        self.datapath = datapath
        self.analpath = analpath
        self.prefix = prefix
    def blockshaped(self,arr,nrows,ncols):
        h, w = arr.shape
        assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
        assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
        return (arr.reshape(h//nrows, nrows, -1, ncols)
                   .swapaxes(1,2)
                   .reshape(-1, nrows, ncols))
    def tile(self):
        dataset = Dataset(self.datapath+self.prefix)
        X = dataset.as_array(stitched=False,axes=['z','channel','row','column'])
        nz,nc,nt,_,nx,ny = X.shape
        print('Tiling Channel 0\n')
        ch0 = np.asarray(X[:,0,:,:,:-self.overlap,:-self.overlap])
        ch0 = np.max(ch0,axis=0) #max intensity projection
        ch0 = ch0.swapaxes(1,2)
        ch0 = ch0.reshape((nt*(nx-self.overlap),nt*(ny-self.overlap)))
        tifffile.imwrite(self.analpath+self.prefix+'/'+self.prefix+'_mxtiled_ch0.tif',ch0)
        del ch0
        print('Tiling Channel 1\n')
        ch1 = np.asarray(X[:,1,:,:,:-self.overlap,:-self.overlap])
        ch1 = np.max(ch1,axis=0) #max intensity projection
        ch1 = ch1.swapaxes(1,2)
        ch1 = ch1.reshape((nt*(nx-self.overlap),nt*(ny-self.overlap)))
        tifffile.imwrite(self.analpath+self.prefix+'/'+self.prefix+'_mxtiled_ch1.tif',ch1)
        del ch1
        print('Tiling Channel 2\n')
        ch2 = np.asarray(X[:,2,:,:,:-self.overlap,:-self.overlap])
        ch2 = np.max(ch2,axis=0) #max intensity projection
        ch2 = ch2.swapaxes(1,2)
        ch2 = ch2.reshape((nt*(nx-self.overlap),nt*(ny-self.overlap)))
        tifffile.imwrite(self.analpath+self.prefix+'/'+self.prefix+'_mxtiled_ch2.tif',ch2)
        del ch2
        print('Tiling Channel 3\n')
        ch3 = np.asarray(X[:,3,:,:,:-self.overlap,:-self.overlap])
        ch3 = np.max(ch3,axis=0) #max intensity projection
        ch3 = ch3.swapaxes(1,2)
        ch3 = ch3.reshape((nt*(nx-self.overlap),nt*(ny-self.overlap)))
        tifffile.imwrite(self.analpath+self.prefix+'/'+self.prefix+'_mxtiled_ch3.tif',ch3)
        del ch3
        print('Tiling Channel 4\n')
        ch4 = np.asarray(X[:,4,:,:,:-self.overlap,:-self.overlap])
        ch4 = np.max(ch4,axis=0) #max intensity projection
        ch4 = ch4.swapaxes(1,2)
        ch4 = ch4.reshape((nt*(nx-self.overlap),nt*(ny-self.overlap)))
        tifffile.imwrite(self.analpath+self.prefix+'/'+self.prefix+'_mxtiled_ch4.tif',ch4)
        del ch4
