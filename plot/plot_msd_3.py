import json
import matplotlib.pyplot as plt
import pandas as pd
import trackpy as tp
import numpy as np
import seaborn as sns
from BaseSMLM.utils import Tracker2D
from plot_msd import *

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

prefixes += [
'231206_Control_646_2pm_1hour_L640_5mW_100ms__11',
'231206_Control_646_2pm_1hour_L640_5mW_100ms__18',
'231206_Control_646_2pm_1hour_L640_5mW_100ms__22',
'231206_Control_646_2pm_1hour_L640_5mW_100ms__23',
'231206_Control_646_2pm_1hour_L640_5mW_100ms__25',
'231206_Control_646_2pm_1hour_L640_5mW_100ms__29',
'231206_Control_646_2pm_1hour_L640_5mW_100ms__37',
'231206_Control_646_2pm_1hour_L640_5mW_100ms__39',
'231206_Control_646_2pm_1hour_L640_5mW_100ms__7'
]

prefixes_jq1 += [
'231206_JQ1_646_2pm_1hour_L640_5mW_100ms__10',
'231206_JQ1_646_2pm_1hour_L640_5mW_100ms__18',
'231206_JQ1_646_2pm_1hour_L640_5mW_100ms__19',
'231206_JQ1_646_2pm_1hour_L640_5mW_100ms__20',
'231206_JQ1_646_2pm_1hour_L640_5mW_100ms__35',
'231206_JQ1_646_2pm_1hour_L640_5mW_100ms__36',
'231206_JQ1_646_2pm_1hour_L640_5mW_100ms__38',
'231206_JQ1_646_2pm_1hour_L640_5mW_100ms__39',
'231206_JQ1_646_2pm_1hour_L640_5mW_100ms__42',
'231206_JQ1_646_2pm_1hour_L640_5mW_100ms__45',
'231206_JQ1_646_2pm_1hour_L640_5mW_100ms__57',
'231206_JQ1_646_2pm_1hour_L640_5mW_100ms__8'
]

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

min_traj_length = 80
with open('plot_msd_1.json', 'r') as f:
    config = json.load(f)

imsds = []
for n,prefix in enumerate(prefixes):
    print("Processing " + prefix)
    tracker = Tracker2D()
    linked = pd.read_csv(config['analpath'] + prefix + '/' + prefix + '_link.csv')
    linked = tp.filter_stubs(linked,min_traj_length)
    imsd = tracker.imsd(linked)
    imsds.append(imsd)
    
imsds = pd.concat(imsds,axis=1)

imsds_jq1 = []
for n,prefix in enumerate(prefixes_jq1):
    print("Processing " + prefix)
    tracker = Tracker2D()
    linked = pd.read_csv(config['analpath'] + prefix + '/' + prefix + '_link.csv')
    linked = tp.filter_stubs(linked,min_traj_length)
    imsd = tracker.imsd(linked)
    imsds_jq1.append(imsd)
    
imsds_jq1 = pd.concat(imsds_jq1,axis=1)

with open('plot_msd_2.json', 'r') as f:
    config = json.load(f)

imsds_wt = []
for n,prefix in enumerate(prefixes_wt):
    print("Processing " + prefix)
    tracker = Tracker2D()
    linked = pd.read_csv(config['analpath'] + 'WT/' + prefix + '/' + prefix + '_link.csv')
    linked = tp.filter_stubs(linked,min_traj_length)
    imsd = tracker.imsd(linked,max_lag=10)
    imsds_wt.append(imsd)
    
imsds_wt = pd.concat(imsds_wt,axis=1,ignore_index=True)
    
imsds_bd = []
for n,prefix in enumerate(prefixes_bd):
    print("Processing " + prefix)
    tracker = Tracker2D()
    linked = pd.read_csv(config['analpath'] + 'BD/' + prefix + '/' + prefix + '_link.csv')
    linked = tp.filter_stubs(linked,min_traj_length)
    imsd = tracker.imsd(linked,max_lag=10)
    imsds_bd.append(imsd)
    
imsds_bd = pd.concat(imsds_bd,axis=1,ignore_index=True)

    
fig,ax=plt.subplots()
plot_msds(imsds,ax,color='black')
#plot_msds(imsds_jq1,ax,color='red')
plot_msds(imsds_wt,ax,color='cyan')
plot_msds(imsds_bd,ax,color='blue')
plt.show()

fig, ax = plt.subplots(figsize=(3.5,3))
plot_avg_msd(imsds,ax,color='black',label='Control')
#plot_avg_msd(imsds_jq1,ax,color='red',label='JQ1')
plot_avg_msd(imsds_wt,ax,color='cyan',label='WT')
plot_avg_msd(imsds_bd,ax,color='blue',label='BD')
plt.show()

nrows = imsds.shape[0]
fig,ax=plt.subplots(2,5,figsize=(10,4))
ax = ax.ravel()
#plot_kde_for_each_row(imsds_jq1,ax,color='red')
plot_kde_for_each_row(imsds,ax,color='black')
plot_kde_for_each_row(imsds_bd,ax,color='blue')
plot_kde_for_each_row(imsds_wt,ax,color='cyan')
plt.tight_layout()
plt.show()

