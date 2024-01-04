import json
import matplotlib.pyplot as plt
import pandas as pd
import trackpy as tp
import numpy as np
from BaseSMLM.utils import Tracker2D

prefixes = [
'231206_Control_646_2pm_1hour_L640_5mW_100ms__10',    
'231206_Control_646_2pm_1hour_L640_5mW_100ms__14',    
'231206_Control_646_2pm_1hour_L640_5mW_100ms__15',    
'231206_Control_646_2pm_1hour_L640_5mW_100ms__17',    
'231206_Control_646_2pm_1hour_L640_5mW_100ms__19',    
'231206_Control_646_2pm_1hour_L640_5mW_100ms__1' ,  
'231206_Control_646_2pm_1hour_L640_5mW_100ms__24',    
'231206_Control_646_2pm_1hour_L640_5mW_100ms__35',    
'231206_Control_646_2pm_1hour_L640_5mW_100ms__3',
'231206_Control_646_2pm_1hour_L640_5mW_100ms__4',  
'231206_Control_646_2pm_1hour_L640_5mW_100ms__6',    
'231206_Control_646_2pm_1hour_L640_5mW_100ms__8',    
'231206_Control_646_2pm_1hour_L640_5mW_100ms__9'    
]

prefixes_jq1 = [
'231206_JQ1_646_2pm_1hour_L640_5mW_100ms__11',
'231206_JQ1_646_2pm_1hour_L640_5mW_100ms__14',    
'231206_JQ1_646_2pm_1hour_L640_5mW_100ms__15',    
'231206_JQ1_646_2pm_1hour_L640_5mW_100ms__16',    
'231206_JQ1_646_2pm_1hour_L640_5mW_100ms__17',    
'231206_JQ1_646_2pm_1hour_L640_5mW_100ms__27',    
'231206_JQ1_646_2pm_1hour_L640_5mW_100ms__32',    
'231206_JQ1_646_2pm_1hour_L640_5mW_100ms__4',  
'231206_JQ1_646_2pm_1hour_L640_5mW_100ms__5',   
'231206_JQ1_646_2pm_1hour_L640_5mW_100ms__7',    
'231206_JQ1_646_2pm_1hour_L640_5mW_100ms__9'   
]

with open('plot_msd_1.json', 'r') as f:
    config = json.load(f)

imsds = []
for n,prefix in enumerate(prefixes):
    print("Processing " + prefix)
    tracker = Tracker2D()
    linked = pd.read_csv(config['analpath'] + prefix + '/' + prefix + '_link.csv')
    linked = tp.filter_stubs(linked,80)
    imsd = tracker.imsd(linked)
    imsds.append(imsd)
    
imsds = pd.concat(imsds,axis=1)

imsds_jq1 = []
for n,prefix in enumerate(prefixes_jq1):
    print("Processing " + prefix)
    tracker = Tracker2D()
    linked = pd.read_csv(config['analpath'] + prefix + '/' + prefix + '_link.csv')
    linked = tp.filter_stubs(linked,80)
    imsd = tracker.imsd(linked)
    imsds_jq1.append(imsd)
    
imsds_jq1 = pd.concat(imsds_jq1,axis=1)
    
"""
fig, ax = plt.subplots()
ax.plot(imsds_bd.index, imsds_bd, 'k-', color='red', alpha=0.1)
ax.plot(imsds_wt.index, imsds_wt, 'k-', color='black', alpha=0.1)
ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
       xlabel='lag time $t$')
"""

fig, ax = plt.subplots(figsize=(3.5,3))
nt,ns = imsds.shape
ctr_stderr = imsds.std(axis=1)/np.sqrt(ns)
ctr_stderr = np.concatenate([np.array([]),ctr_stderr])
tau = np.concatenate([np.array([]),imsds.index])
avg_msd = imsds.mean(axis=1)
avg_msd = np.concatenate([np.array([]),avg_msd])
ax.errorbar(tau, avg_msd, yerr=ctr_stderr,color='black',marker='o',capsize=3.0,alpha=0.5,label='Control')

nt,ns = imsds_jq1.shape
jq1_stderr = imsds_jq1.std(axis=1)/np.sqrt(ns)
jq1_stderr = np.concatenate([np.array([]),jq1_stderr])
tau = np.concatenate([np.array([]),imsds_jq1.index])
avg_msd = imsds_jq1.mean(axis=1)
avg_msd = np.concatenate([np.array([]),avg_msd])
ax.errorbar(tau, avg_msd, yerr=jq1_stderr,color='red',marker='o',capsize=3.0,alpha=0.5,label='JQ1')
ax.set(ylabel=r'$\log_{10}\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
       xlabel=r'$\log_{10}\tau$ (sec)')
       
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(loc='upper left')
plt.tight_layout()
plt.show()

