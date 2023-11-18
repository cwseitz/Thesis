import numpy as np
import matplotlib.pyplot as plt
from cayenne.simulation import Simulation
from SMLM.mains.run.pipes import *
from SMLM.torch.dataset import SMLMDataset
from matplotlib.gridspec import GridSpec

class Figure_6:
    """Grid-search inference of photoswitching constants under a two-state model"""
    def __init__(self,config):
        self.config = config
        
    def plot(self,prefix,nn=100):

        dataset = SMLMDataset(self.config['datapath']+prefix,prefix)
        self.pipe = PipelineLifetime(self.config)   
        adu, binary = self.pipe.forward(dataset.stack,thresh=self.config['thresh'],
                                        det_idx=self.config['det_idx'],ax=None)
        avgs = np.mean(adu,axis=1)
 
        k12vec, k21vec, map = self.get_map(nn)
        idx = self.avg_to_map_idx(avgs,map)      

        fig,ax = plt.subplots()
        ax.imshow(map,cmap='Greys')
        xticks = np.arange(0,nn,int(round(nn/5)))
        xticklabels = ["{:6.2f}".format(k21vec[i]) for i in xticks]
        yticks = np.arange(0,nn,int(round(nn/5)))
        yticklabels = ["{:6.2f}".format(k12vec[i]) for i in yticks]
        ax.scatter(idx[:,1],idx[:,0],color='cyan',marker='x',s=5)
        ax.invert_yaxis()
        ax.set_xlabel(r'$k_{21}\; (\mathrm{frames}^{-1})$',fontsize=10)
        ax.set_ylabel(r'$k_{12}\; (\mathrm{frames}^{-1})$',fontsize=10)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        plt.tight_layout()
        plt.show()
        
    def add_time_series(self,ax1,ax2,adu,binary):
        time = np.linspace(0,2,adu.shape[0])
        idx1 = 0; idx2 = 1
        ax1.plot(time,adu[:,idx1],color='cornflowerblue')
        ax1.plot(time,binary[:,idx1],color='red',alpha=0.3)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Intensity (a.u.)')
        ax2.plot(time,adu[:,idx2],color='cornflowerblue')
        ax2.plot(time,binary[:,idx2],color='red',alpha=0.3)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Intensity (a.u.)')
        
    def avg_to_map_idx(self,avgs,map):
        idx = []
        for avg in avgs:
            A = np.abs(map-avg)
            this_idx = np.unravel_index(A.argmin(), A.shape)
            idx.append(this_idx)
        idx = np.array(idx)
        return idx
        
    def get_map(self,nn=20,k12min=0.01,k12max=200.0,k21min=0.01,k21max=200.0):
        k12vec = np.linspace(k12min,k12max,nn)
        k21vec = np.linspace(k21min,k21max,nn)
        map = self.map_ssa(k12vec,k21vec)
        return k12vec, k21vec, map
        
    def map_ssa(self,k12vec,k21vec,plot=False):
    
        def ssa(k12,k21,plot=False,tmax=1.0):

            model_str = """
                const compartment comp1;
                comp1 = 1.0; # volume of compartment

                r1: A => B; k12;
                r2: B => A; k21;

                k12 = {};
                k21 = {};
                chem_flag = false;

                A = 1;
                B = 0;
            """.format(k12,k21)
            
            sim = Simulation.load_model(model_str, "ModelString")
            sim.simulate(max_t=tmax, max_iter=10000, n_rep=1000, algorithm="direct")
            results = sim.results
            otimes = []
            for x, t, status in results:
                x = x[:,0] #on state
                t = np.append(t,tmax)
                diff = np.diff(t)
                otime = np.sum(x*diff)
                otimes.append(otime)

            otimes = np.array(otimes)

            if plot:
                bin_size = 0.1
                bins = np.arange(0,1,bin_size)
                vals, bins = np.histogram(otimes,bins=bins,density=True)
                fig,ax=plt.subplots()
                ax.set_title(f'k12={k12},k21={k21}')
                ax.bar(bins[:-1], vals, width=bin_size, align='edge',alpha=0.5)
                plt.show()
                
            return otimes
                
        X,Y = np.meshgrid(k12vec,k21vec,indexing='ij')
        nx,ny = len(k12vec), len(k21vec)
        avg_map = np.zeros((nx,ny))
        for i in range(nx):
            for j in range(ny):
                times = ssa(X[i,j],Y[i,j],plot=plot)
                avg_map[i,j] = np.mean(times)
        return avg_map
