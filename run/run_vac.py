from skimage.io import imsave
from BaseSMLM.utils.dataset import SMLMDataset
from BaseSMLM.utils import Tracker2D, make_animation
import matplotlib.pyplot as plt
import trackpy as tp
import napari
import pandas as pd
import numpy as np
import json

prefixes_7a = [
'240118_SMT_7A_JF646_1.5pm_1h_L640_5mW_100ms___1',
'240118_SMT_7A_JF646_1.5pm_1h_L640_5mW_100ms___10',
'240118_SMT_7A_JF646_1.5pm_1h_L640_5mW_100ms___11',
'240118_SMT_7A_JF646_1.5pm_1h_L640_5mW_100ms___13',
'240118_SMT_7A_JF646_1.5pm_1h_L640_5mW_100ms___14',
'240118_SMT_7A_JF646_1.5pm_1h_L640_5mW_100ms___15',
'240118_SMT_7A_JF646_1.5pm_1h_L640_5mW_100ms___17',
'240118_SMT_7A_JF646_1.5pm_1h_L640_5mW_100ms___19',
'240118_SMT_7A_JF646_1.5pm_1h_L640_5mW_100ms___2',
'240118_SMT_7A_JF646_1.5pm_1h_L640_5mW_100ms___21',
'240118_SMT_7A_JF646_1.5pm_1h_L640_5mW_100ms___23',
'240118_SMT_7A_JF646_1.5pm_1h_L640_5mW_100ms___24',
'240118_SMT_7A_JF646_1.5pm_1h_L640_5mW_100ms___29',
'240118_SMT_7A_JF646_1.5pm_1h_L640_5mW_100ms___31',
'240118_SMT_7A_JF646_1.5pm_1h_L640_5mW_100ms___32',
'240118_SMT_7A_JF646_1.5pm_1h_L640_5mW_100ms___33',
'240118_SMT_7A_JF646_1.5pm_1h_L640_5mW_100ms___34',
'240118_SMT_7A_JF646_1.5pm_1h_L640_5mW_100ms___4',
'240118_SMT_7A_JF646_1.5pm_1h_L640_5mW_100ms___7'
]

prefixes_7d = [
'240119_SMT_7D_JF646_1.5pm_1h_L640_5mW_100ms___12',
'240119_SMT_7D_JF646_1.5pm_1h_L640_5mW_100ms___3',
'240119_SMT_7D_JF646_1.5pm_1h_L640_5mW_100ms___4',
'240119_SMT_7D_JF646_1.5pm_1h_L640_5mW_100ms___6',
'240119_SMT_7D_JF646_1.5pm_1h_L640_5mW_100ms___7',
'240121_SMT_7D_JF646_1.5pm_1h_L640_5mW_100m_10',
'240121_SMT_7D_JF646_1.5pm_1h_L640_5mW_100m_12',
'240121_SMT_7D_JF646_1.5pm_1h_L640_5mW_100m_13',
'240121_SMT_7D_JF646_1.5pm_1h_L640_5mW_100m_15',
'240121_SMT_7D_JF646_1.5pm_1h_L640_5mW_100m_16',
'240121_SMT_7D_JF646_1.5pm_1h_L640_5mW_100m_17',
'240121_SMT_7D_JF646_1.5pm_1h_L640_5mW_100m_18',
'240121_SMT_7D_JF646_1.5pm_1h_L640_5mW_100m_19',
'240121_SMT_7D_JF646_1.5pm_1h_L640_5mW_100m_2',
'240121_SMT_7D_JF646_1.5pm_1h_L640_5mW_100m_21',
'240121_SMT_7D_JF646_1.5pm_1h_L640_5mW_100m_22',
'240121_SMT_7D_JF646_1.5pm_1h_L640_5mW_100m_23',
'240121_SMT_7D_JF646_1.5pm_1h_L640_5mW_100m_24',
'240121_SMT_7D_JF646_1.5pm_1h_L640_5mW_100m_25',
'240121_SMT_7D_JF646_1.5pm_1h_L640_5mW_100m_26',
'240121_SMT_7D_JF646_1.5pm_1h_L640_5mW_100m_27',
'240121_SMT_7D_JF646_1.5pm_1h_L640_5mW_100m_3',
'240121_SMT_7D_JF646_1.5pm_1h_L640_5mW_100m_4',
'240121_SMT_7D_JF646_1.5pm_1h_L640_5mW_100m_5',
'240121_SMT_7D_JF646_1.5pm_1h_L640_5mW_100m_7',
'240121_SMT_7D_JF646_1.5pm_1h_L640_5mW_100m_8',
'240121_SMT_7D_JF646_1.5pm_1h_L640_5mW_100m_9'
]

prefixes_WT = [
'231226_WT_5mw_100ms_JF646_2pm_1hour__15',
#'231226_WT_5mw_100ms_JF646_2pm_1hour__16', #few traj
'231226_WT_5mw_100ms_JF646_2pm_1hour__23',
'231226_WT_5mw_100ms_JF646_2pm_1hour__24',
'231226_WT_5mw_100ms_JF646_2pm_1hour__26',
'231226_WT_5mw_100ms_JF646_2pm_1hour__33',
'231226_WT_5mw_100ms_JF646_2pm_1hour__3'
]


prefixes_WT += [
'231226_WT_5mw_100ms_JF646_2pm_1hour__11',
'231226_WT_5mw_100ms_JF646_2pm_1hour__14',
#'231226_WT_5mw_100ms_JF646_2pm_1hour__19', #few traj
'231226_WT_5mw_100ms_JF646_2pm_1hour__21',
#'231226_WT_5mw_100ms_JF646_2pm_1hour__29', #few traj
'231226_WT_5mw_100ms_JF646_2pm_1hour__30',
'231226_WT_5mw_100ms_JF646_2pm_1hour__32',
'231226_WT_5mw_100ms_JF646_2pm_1hour__5',
'231226_WT_5mw_100ms_JF646_2pm_1hour__7',
'231226_WT_5mw_100ms_JF646_2pm_1hour__8' 
]

prefixes_BD = [
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

prefixes_BD += [
'231226_BD_5mw_100ms_JF646_2pm_1hour__14',
'231226_BD_5mw_100ms_JF646_2pm_1hour__18',
'231226_BD_5mw_100ms_JF646_2pm_1hour__20',
'231226_BD_5mw_100ms_JF646_2pm_1hour__21',
'231226_BD_5mw_100ms_JF646_2pm_1hour__3',
'231226_BD_5mw_100ms_JF646_2pm_1hour__33',
'231226_BD_5mw_100ms_JF646_2pm_1hour__34',
#'231226_BD_5mw_100ms_JF646_2pm_1hour__35', #few traj
#'231226_BD_5mw_100ms_JF646_2pm_1hour__42', #few traj
#'231226_BD_5mw_100ms_JF646_2pm_1hour__5', #few traj
'231226_BD_5mw_100ms_JF646_2pm_1hour__9'
]



with open('run_vac.json', 'r') as f:
    config = json.load(f)
   
   
def vacf(spots,nauto=10,plot_ind=False):
    tracker = Tracker2D()
    linked = tracker.link(spots,memory=0) #take only continuous trajectories
    linked = tp.filter_stubs(linked,30)
    linked = linked[['x_mle','y_mle','frame','particle']]
    linked = linked.rename(columns={'x_mle':'x','y_mle':'y'})
    frame_nums = np.arange(0,199,1)
    X = []
    for frame in frame_nums:
        j = tp.motion.relate_frames(linked,frame,frame+1)
        j = j.assign(frame=frame); X.append(j)
    X = pd.concat(X,axis=0)
    X = X.reset_index()
    autos = []
    for particle in X['particle'].unique():
        X0 = X.loc[X['particle'] == particle]
        dx = X0['dx'].values[:-1]
        dy = X0['dy'].values[:-1]
        xauto = np.correlate(dx,dx,mode='same')
        yauto = np.correlate(dy,dy,mode='same')
        auto = xauto[len(xauto)//2:]
        auto += yauto[len(yauto)//2:]
        auto = auto/auto[0] #normd
        autos.append(auto[:nauto])
        if plot_ind:
            fig,ax=plt.subplots(1,2)
            ax[0].plot(dx,color='black')
            ax[0].plot(dy,color='blue')
            ax[1].plot(auto[:nauto],color='red')
            plt.show()
    autos = np.array(autos)
    return autos


fig,ax=plt.subplots()
autos_7d = []
for prefix in prefixes_7d:
    print("Processing " + prefix)
    #dataset = SMLMDataset(config['datapath']+prefix,prefix)
    spots = pd.read_csv(config['analpath1'] + prefix + '/' + prefix + '_spots.csv')
    spots = spots.loc[spots['x_err'] < 0.03]
    #make_animation(dataset.stack,spots)
    autos = vacf(spots,plot_ind=False)
    autos_7d.append(autos)
    #ax.plot(autos.T,color='red',alpha=0.1)
    
bins = np.linspace(-0.7,-0.1,10)
autos_7d = np.concatenate(autos_7d,axis=0)
vals,bins = np.histogram(autos_7d[:,1],bins=bins,density=True)
ax.plot(bins[:-1],vals,marker='o',color='red')
   
autos_WT = []
for prefix in prefixes_WT:
    print("Processing " + prefix)
    #dataset = SMLMDataset(config['datapath']+prefix,prefix)
    spots = pd.read_csv(config['analpath2'] + prefix + '/' + prefix + '_spots.csv')
    spots = spots.loc[spots['x_err'] < 0.03]
    #make_animation(dataset.stack,spots)
    autos = vacf(spots,plot_ind=False)
    autos_WT.append(autos)
    #ax.plot(autos.T,color='blue',alpha=0.1)
    
bins = np.linspace(-0.7,-0.1,10)
autos_WT = np.concatenate(autos_WT,axis=0)
vals,bins = np.histogram(autos[:,1],bins=bins,density=True)
ax.plot(bins[:-1],vals,marker='o',color='blue')
    
plt.show()    
