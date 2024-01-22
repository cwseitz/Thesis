import json
import matplotlib.pyplot as plt
import pandas as pd
import trackpy as tp
import numpy as np
import seaborn as sns
from BaseSMLM.utils import Tracker2D
from DeepSMLM.torch.dataset import SMLMDataset
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


prefixes_4h = [
'240114_JQ1_4hours directly_646_2pm_1hour_L640_5mW_100ms___10',
'240114_JQ1_4hours directly_646_2pm_1hour_L640_5mW_100ms___16',
'240114_JQ1_4hours directly_646_2pm_1hour_L640_5mW_100ms___19',
'240114_JQ1_4hours directly_646_2pm_1hour_L640_5mW_100ms___22',
'240114_JQ1_4hours directly_646_2pm_1hour_L640_5mW_100ms___25',
'240114_JQ1_4hours directly_646_2pm_1hour_L640_5mW_100ms___28',
'240114_JQ1_4hours directly_646_2pm_1hour_L640_5mW_100ms___2',
'240114_JQ1_4hours directly_646_2pm_1hour_L640_5mW_100ms___30'
]

prefixes_8h = [
'240114_JQ1_8hours directly_646_2pm_1hour_L640_5mW_100ms___11',
'240114_JQ1_8hours directly_646_2pm_1hour_L640_5mW_100ms___13',
'240114_JQ1_8hours directly_646_2pm_1hour_L640_5mW_100ms___15',
'240114_JQ1_8hours directly_646_2pm_1hour_L640_5mW_100ms___16',
'240114_JQ1_8hours directly_646_2pm_1hour_L640_5mW_100ms___18',
'240114_JQ1_8hours directly_646_2pm_1hour_L640_5mW_100ms___19',
#'240114_JQ1_8hours directly_646_2pm_1hour_L640_5mW_100ms___27',
'240114_JQ1_8hours directly_646_2pm_1hour_L640_5mW_100ms___3',
'240114_JQ1_8hours directly_646_2pm_1hour_L640_5mW_100ms___5'
]

prefixes_16h = [
'240115_JQ1_16hours directly_646_2pm_1hour_L640_5mW_100ms___14',
'240115_JQ1_16hours directly_646_2pm_1hour_L640_5mW_100ms___16',
'240115_JQ1_16hours directly_646_2pm_1hour_L640_5mW_100ms___18',
'240115_JQ1_16hours directly_646_2pm_1hour_L640_5mW_100ms___1',
'240115_JQ1_16hours directly_646_2pm_1hour_L640_5mW_100ms___23',
'240115_JQ1_16hours directly_646_2pm_1hour_L640_5mW_100ms___25',
'240115_JQ1_16hours directly_646_2pm_1hour_L640_5mW_100ms___26',
'240115_JQ1_16hours directly_646_2pm_1hour_L640_5mW_100ms___28',
'240115_JQ1_16hours directly_646_2pm_1hour_L640_5mW_100ms___2'
]

min_traj_length = 80

with open('plot_msd_1.json', 'r') as f:
    config = json.load(f)

imsds = []
for n,prefix in enumerate(prefixes):
    print("Processing " + prefix)
    #dataset = SMLMDataset(config['datapath']+prefix,prefix)
    tracker = Tracker2D()
    spots = pd.read_csv(config['analpath'] + prefix + '/' + prefix + '_spots.csv')
    linked = tracker.link(spots)
    linked = tp.filter_stubs(linked,min_traj_length)
    #tp.plot_traj(linked,superimpose=dataset.stack[0],
    #             pos_columns=['y_mle','x_mle'])
    imsd = tracker.imsd(linked)
    imsds.append(imsd)
    
imsds = pd.concat(imsds,axis=1)

imsds_jq1 = []
for n,prefix in enumerate(prefixes_jq1):
    print("Processing " + prefix)
    #dataset = SMLMDataset(config['datapath']+prefix,prefix)
    tracker = Tracker2D()
    spots = pd.read_csv(config['analpath'] + prefix + '/' + prefix + '_spots.csv')
    linked = tracker.link(spots)
    linked = tp.filter_stubs(linked,min_traj_length)
    #tp.plot_traj(linked,superimpose=dataset.stack[0],
    #             pos_columns=['y_mle','x_mle'])
    imsd = tracker.imsd(linked)
    imsds_jq1.append(imsd)
    
imsds_jq1 = pd.concat(imsds_jq1,axis=1)
    

with open('plot_msd_4.json', 'r') as f:
    config = json.load(f)

imsds_4h = []
for n,prefix in enumerate(prefixes_4h):
    print("Processing " + prefix)
    dataset = SMLMDataset(config['datapath']+prefix,prefix)
    tracker = Tracker2D()
    spots = pd.read_csv(config['analpath'] + prefix + '/' + prefix + '_spots.csv')
    linked = tracker.link(spots)
    linked = tp.filter_stubs(linked,min_traj_length)
    #tp.plot_traj(linked,superimpose=dataset.stack[0],
    #             pos_columns=['y_mle','x_mle'])
    imsd = tracker.imsd(linked)
    imsds_4h.append(imsd)
    
imsds_4h = pd.concat(imsds_4h,axis=1)

imsds_8h = []
for n,prefix in enumerate(prefixes_8h):
    print("Processing " + prefix)
    dataset = SMLMDataset(config['datapath'].replace('4h','8h')+prefix,prefix)
    tracker = Tracker2D()
    spots = pd.read_csv(config['analpath'] + prefix + '/' + prefix + '_spots.csv')
    linked = tracker.link(spots)
    linked = tp.filter_stubs(linked,min_traj_length)
    #tp.plot_traj(linked,superimpose=dataset.stack[0],
    #             pos_columns=['y_mle','x_mle'])
    imsd = tracker.imsd(linked)
    imsds_8h.append(imsd)
    
imsds_8h = pd.concat(imsds_8h,axis=1)

imsds_16h = []
for n,prefix in enumerate(prefixes_16h):
    print("Processing " + prefix)
    dataset = SMLMDataset(config['datapath'].replace('4h','16h')+prefix,prefix)
    tracker = Tracker2D()
    spots = pd.read_csv(config['analpath'] + prefix + '/' + prefix + '_spots.csv')
    linked = tracker.link(spots)
    linked = tp.filter_stubs(linked,min_traj_length)
    #tp.plot_traj(linked,superimpose=dataset.stack[0],
    #             pos_columns=['y_mle','x_mle'])
    imsd = tracker.imsd(linked)
    imsds_16h.append(imsd)
    
imsds_16h = pd.concat(imsds_16h,axis=1)
    
    
fig,ax=plt.subplots()
plot_msds(imsds,ax,color='red')
plot_msds(imsds_jq1,ax,color='green')
plot_msds(imsds_4h,ax,color='black')
plot_msds(imsds_8h,ax,color='blue')
plot_msds(imsds_16h,ax,color='cyan')
plt.show()

fig, ax = plt.subplots(figsize=(3.5,3))
plot_avg_msd(imsds,ax,color='red',label='Control')
plot_avg_msd(imsds_jq1,ax,color='green',label='8h (gapped)')
plot_avg_msd(imsds_4h,ax,color='black',label='4h')
plot_avg_msd(imsds_8h,ax,color='blue',label='8h')
plot_avg_msd(imsds_16h,ax,color='cyan',label='16h')
plt.show()

nrows = imsds_4h.shape[0]
fig,ax=plt.subplots(2,5,figsize=(10,4))
ax = ax.ravel()
plot_kde_for_each_row(imsds,ax,color='red')
plot_kde_for_each_row(imsds_jq1,ax,color='green')
plot_kde_for_each_row(imsds_4h,ax,color='black')
plot_kde_for_each_row(imsds_8h,ax,color='blue')
plot_kde_for_each_row(imsds_16h,ax,color='cyan')
plt.tight_layout()
plt.show()

