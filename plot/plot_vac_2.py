import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import trackpy as tp
from BaseSMLM.utils import Tracker2D

prefixes_wt = [
'231226_WT_5mw_100ms_JF646_2pm_1hour__11',
'231226_WT_5mw_100ms_JF646_2pm_1hour__14',
'231226_WT_5mw_100ms_JF646_2pm_1hour__19',
'231226_WT_5mw_100ms_JF646_2pm_1hour__21',
'231226_WT_5mw_100ms_JF646_2pm_1hour__29',
'231226_WT_5mw_100ms_JF646_2pm_1hour__30',
'231226_WT_5mw_100ms_JF646_2pm_1hour__32',
'231226_WT_5mw_100ms_JF646_2pm_1hour__5',
'231226_WT_5mw_100ms_JF646_2pm_1hour__7',
'231226_WT_5mw_100ms_JF646_2pm_1hour__8' 
]

prefixes_bd = [
'231226_BD_5mw_100ms_JF646_2pm_1hour__14',
'231226_BD_5mw_100ms_JF646_2pm_1hour__18',
'231226_BD_5mw_100ms_JF646_2pm_1hour__20',
'231226_BD_5mw_100ms_JF646_2pm_1hour__21',
'231226_BD_5mw_100ms_JF646_2pm_1hour__3',
'231226_BD_5mw_100ms_JF646_2pm_1hour__33',
'231226_BD_5mw_100ms_JF646_2pm_1hour__34',
'231226_BD_5mw_100ms_JF646_2pm_1hour__35',
'231226_BD_5mw_100ms_JF646_2pm_1hour__42',
'231226_BD_5mw_100ms_JF646_2pm_1hour__5',
'231226_BD_5mw_100ms_JF646_2pm_1hour__9'
]

with open('plot_vac_2.json', 'r') as f:
    config = json.load(f)

imsds_wt = []
for n,prefix in enumerate(prefixes_wt):
    print("Processing " + prefix)
    tracker = Tracker2D()
    linked = pd.read_csv(config['analpath'] + 'WT/' + prefix + '/' + prefix + '_link.csv')
    linked = tp.filter_stubs(linked,60)
    pos = linked.set_index(['frame', 'particle'])['x_mle'].unstack()
    print(pos)
    imsd = tracker.imsd(linked,max_lag=10)
    imsds_wt.append(imsd)
    
imsds_wt = pd.concat(imsds_wt,axis=1,ignore_index=True)
    
imsds_bd = []
for n,prefix in enumerate(prefixes_bd):
    print("Processing " + prefix)
    tracker = Tracker2D()
    linked = pd.read_csv(config['analpath'] + 'BD/' + prefix + '/' + prefix + '_link.csv')
    linked = tp.filter_stubs(linked,60)
    imsd = tracker.imsd(linked,max_lag=10)
    imsds_bd.append(imsd)
    
imsds_bd = pd.concat(imsds_bd,axis=1,ignore_index=True)

"""
fig, ax = plt.subplots()
ax.plot(imsds_bd.index, imsds_bd, 'k-', color='red', alpha=0.1)
ax.plot(imsds_wt.index, imsds_wt, 'k-', color='black', alpha=0.1)
ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
       xlabel='lag time $t$')
"""

fig, ax = plt.subplots(figsize=(3.5,3))
nt,ns = imsds_bd.shape
bd_stderr = imsds_bd.std(axis=1)/np.sqrt(ns)
bd_stderr = np.concatenate([np.array([]),bd_stderr])
tau = np.concatenate([np.array([]),imsds_bd.index])
avg_msd = imsds_bd.mean(axis=1)
avg_msd = np.concatenate([np.array([]),avg_msd])
ax.errorbar(tau, avg_msd, yerr=bd_stderr,color='red',marker='o',capsize=3.0,alpha=0.5,label='BD')

nt,ns = imsds_wt.shape
wt_stderr = imsds_wt.std(axis=1)/np.sqrt(ns)
wt_stderr = np.concatenate([np.array([]),wt_stderr])
tau = np.concatenate([np.array([]),imsds_wt.index])
avg_msd = imsds_wt.mean(axis=1)
avg_msd = np.concatenate([np.array([]),avg_msd])
ax.errorbar(tau, avg_msd, yerr=wt_stderr,color='black',marker='o',capsize=3.0,alpha=0.5,label='WT')
ax.set(ylabel=r'$\log_{10}\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
       xlabel=r'$\log_{10}\tau$ (sec)')
       
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(loc='upper left')
plt.tight_layout()
plt.show()

