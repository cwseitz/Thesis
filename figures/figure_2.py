import numpy as np
import matplotlib.pyplot as plt
from SMLM.torch.dataset import SMLMDataset
from SMLM.mains.run.pipes import PipelineMLE3D_MCMC

class Figure_2:
    """MCMC to estimate localization uncertainty in 3D"""
    def __init__(self,config):
        self.config = config

    def plot(self,prefix):
        """Just going to rely on some diagnostic plots for this one"""
        dataset = SMLMDataset(self.config['datapath']+prefix,prefix)
        dataset.stack = np.squeeze(dataset.stack)
        pipe = PipelineMLE3D_MCMC(self.config,dataset)
        pipe.localize(plot=True)
