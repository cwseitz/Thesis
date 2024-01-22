import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import trackpy as tp
import seaborn as sns
from BaseSMLM.utils import Tracker2D
from plot_msd import *

def plot_kde_for_each_row(dataframe,ax,color='red'):
    sns.set(style="whitegrid")
    rows = dataframe.index
    for n,row in enumerate(rows):
        sns.kdeplot(dataframe.loc[row,:], ax=ax[n], color=color)
        ax[n].set_xlabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]')
        ax[n].set_title(r'$\tau$='+f'{rows[n]} sec')
        

prefixes_bd = [
'231226_BD_5mw_100ms_JF646_2pm_1hour__16',
'231226_BD_5mw_100ms_JF646_2pm_1hour__17',
'231226_BD_5mw_100ms_JF646_2pm_1hour__25',
'231226_BD_5mw_100ms_JF646_2pm_1hour__29',
'231226_BD_5mw_100ms_JF646_2pm_1hour__30',
'231226_BD_5mw_100ms_JF646_2pm_1hour__38',
'231226_BD_5mw_100ms_JF646_2pm_1hour__39',
'231226_BD_5mw_100ms_JF646_2pm_1hour__40',
'231226_BD_5mw_100ms_JF646_2pm_1hour__8'
]

prefixes_wt = [
'231226_WT_5mw_100ms_JF646_2pm_1hour__15',
'231226_WT_5mw_100ms_JF646_2pm_1hour__16',
'231226_WT_5mw_100ms_JF646_2pm_1hour__23',
'231226_WT_5mw_100ms_JF646_2pm_1hour__24',
'231226_WT_5mw_100ms_JF646_2pm_1hour__26',
'231226_WT_5mw_100ms_JF646_2pm_1hour__33',
'231226_WT_5mw_100ms_JF646_2pm_1hour__3'
]

prefixes_wt += [
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

prefixes_bd += [
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

with open('plot_msd_2.json', 'r') as f:
    config = json.load(f)

imsds_wt = []
for n,prefix in enumerate(prefixes_wt):
    print("Processing " + prefix)
    tracker = Tracker2D()
    linked = pd.read_csv(config['analpath'] + 'WT/' + prefix + '/' + prefix + '_link.csv')
    linked = tp.filter_stubs(linked,80)
    imsd = tracker.imsd(linked,max_lag=10)
    imsds_wt.append(imsd)
    
imsds_wt = pd.concat(imsds_wt,axis=1,ignore_index=True)
    
imsds_bd = []
for n,prefix in enumerate(prefixes_bd):
    print("Processing " + prefix)
    tracker = Tracker2D()
    linked = pd.read_csv(config['analpath'] + 'BD/' + prefix + '/' + prefix + '_link.csv')
    linked = tp.filter_stubs(linked,80)
    imsd = tracker.imsd(linked,max_lag=10)
    imsds_bd.append(imsd)
    
imsds_bd = pd.concat(imsds_bd,axis=1,ignore_index=True)

fig,ax=plt.subplots()
plot_msds(imsds_wt,ax,color='black')
plot_msds(imsds_bd,ax,color='blue')
plt.show()

fig, ax = plt.subplots(figsize=(3.5,3))
plot_avg_msd(imsds_wt,ax,color='black',label='WT')
plot_avg_msd(imsds_bd,ax,color='blue',label='BD')
plt.show()

nrows = imsds_wt.shape[0]
fig,ax=plt.subplots(2,5,figsize=(10,4))
ax = ax.ravel()
plot_kde_for_each_row(imsds_bd,ax,color='blue')
plot_kde_for_each_row(imsds_wt,ax,color='black')
plt.tight_layout()
plt.show()
