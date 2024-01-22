import json
import matplotlib.pyplot as plt
import pandas as pd
import trackpy as tp
import numpy as np
import seaborn as sns
from BaseSMLM.utils import Tracker2D

def plot_kde_for_each_row(dataframe,ax,color='red'):
    sns.set(style="whitegrid")
    rows = dataframe.index
    for n,row in enumerate(rows):
        sns.kdeplot(dataframe.loc[row,:], ax=ax[n], alpha=0.6, color=color)
        ax[n].set_xlabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]')
        ax[n].set_title(r'$\tau$='+f'{rows[n]} sec')
        
def plot_msds(imsds,ax,color='red'):
    ax.plot(imsds.index, imsds, 'k-', color=color, alpha=0.1)
    ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
           xlabel='lag (sec)')

def plot_avg_msd(imsds,ax,color='red',label=None):
    nt,ns = imsds.shape
    std = imsds.std(axis=1)
    stderr = std/np.sqrt(ns)
    stderr = np.concatenate([np.array([]),stderr])
    tau = np.concatenate([np.array([]),imsds.index])
    avg_msd = imsds.mean(axis=1)
    avg_msd = np.concatenate([np.array([]),avg_msd])
    ax.errorbar(tau, avg_msd,yerr=stderr,color=color,marker='o',capsize=3.0,alpha=0.5,label=label)
    ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
           xlabel='lag (sec)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim([8e-3,2e-2])
    ax.set_yticks(np.linspace(8e-3,1.8e-2,5))
    ax.legend(loc='upper left')
    plt.tight_layout()

