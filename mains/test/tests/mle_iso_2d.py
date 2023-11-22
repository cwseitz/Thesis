import numpy as np
import matplotlib.pyplot as plt
from BaseSMLM.generators import Iso2D
from BaseSMLM.psf.psf2d import *
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

class MLE2DIso_Test:
    """Test a single instance of maximum likelihood estimation"""
    def __init__(self,config):
        self.config = config
        self.cmos_params = [config['eta'],config['texp'],
                            np.load(config['gain'])['arr_0'],
                            np.load(config['offset'])['arr_0'],
                            np.load(config['var'])['arr_0']]  
    def test(self):
        gain,offset,var = self.cmos_params[2:]
        nx,ny = offset.shape
        x0 = nx // 2; y0 = ny // 2
        self.thetagt = np.array([x0,y0,self.config['sigma'],
                                 self.config['N0']])
        iso2d = Iso2D(self.thetagt,self.config)
        adu = iso2d.generate(plot=True)
        adu = adu - offset
        adu = np.clip(adu,0,None)
        lr = np.array([0.001,0.001,0.0,100.0]) #hyperpar
        theta0 = np.zeros_like(self.thetagt)
        theta0 += self.thetagt
        theta0[0] += np.random.normal(0,1)
        theta0[1] += np.random.normal(0,1)
        theta0[3] += 100
        opt = MLE2D(theta0,adu,self.config,theta_gt=self.thetagt)
        theta,loglike,conv = opt.optimize(max_iters=100,lr=lr,plot_fit=True)
        run_mcmc(theta,adu,self.cmos_params,num_samples=1000,warmup_steps=200)

def _lamx(X, x0, sigma):
    alpha = torch.tensor(2.0).sqrt() * sigma
    X = torch.tensor(X)
    return 0.5 * (torch.erf((X + 0.5 - x0) / alpha) - torch.erf((X - 0.5 - x0) / alpha))

def _lamy(Y, y0, sigma):
    alpha = torch.tensor(2.0).sqrt() * sigma
    Y = torch.tensor(Y)
    return 0.5 * (torch.erf((Y + 0.5 - y0) / alpha) - torch.erf((Y - 0.5 - y0) / alpha))

def loglike(theta,adu,cmos_params):
    eta,texp,gain,offset,var = cmos_params
    eta = torch.tensor(eta); texp = torch.tensor(texp)
    gain = torch.tensor(gain); offset = torch.tensor(offset)
    var = torch.tensor(var)
    nx,ny = offset.shape
    x0,y0 = theta; sigma,N0 = 0.92,1000
    X,Y = np.meshgrid(np.arange(0,nx),np.arange(0,ny))
    lam = _lamx(X,x0,sigma)*_lamy(Y,y0,sigma)
    i0 = gain*eta*texp*N0
    muprm = i0*lam + var
    stirling = adu * torch.nan_to_num(torch.log(adu)) - adu
    p = adu*torch.log(muprm)
    p = torch.nan_to_num(p)
    nll = stirling + muprm - p
    nll = torch.sum(nll)
    return nll

def model(adu,cmos_params):
    nx,ny = adu.shape
    lower_bounds = torch.tensor([8.0,8.0])
    upper_bounds = torch.tensor([12.0,12.0])
    theta = pyro.sample('theta', dist.Uniform(lower_bounds, upper_bounds))
    likelihood_value = loglike(theta, adu, cmos_params)
    pyro.sample('obs', dist.Normal(likelihood_value, 1.0), obs=torch.tensor(0.0))

def run_mcmc(theta0,adu,cmos_params,num_samples=1000,warmup_steps=500):
    adu = torch.tensor(adu)
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc.run(adu, cmos_params)
    trace = mcmc.get_samples()
    plot_mcmc_samples(trace)

def plot_mcmc_samples(trace):
    theta = trace['theta'].numpy()
    fig, ax = plt.subplots(1,4)
    nsamples,ntheta = theta.shape
    bins = 30
    for n in range(ntheta):
       ax[n].hist(theta[:,n],bins=bins,color='black',density=True)
    plt.tight_layout()
    plt.show()
 

       
