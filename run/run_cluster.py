from BaseSMLM.utils import RTCluster, RTGridSearch
from skimage.io import imread
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

prefixes_7a_1 = [
#'240119_SMT_7a_JF646_2.5pm_overnight_L640_30mW_10ms___21', #drift
#'240119_SMT_7a_JF646_2.5pm_overnight_L640_30mW_10ms___22', #drift
'240119_SMT_7a_JF646_2.5pm_overnight_L640_30mW_10ms___23',
'240119_SMT_7a_JF646_2.5pm_overnight_L640_30mW_10ms___27',
'240119_SMT_7a_JF646_2.5pm_overnight_L640_30mW_10ms___28',
'240119_SMT_7a_JF646_2.5pm_overnight_L640_30mW_10ms___2',
'240119_SMT_7a_JF646_2.5pm_overnight_L640_30mW_10ms___32',
'240119_SMT_7a_JF646_2.5pm_overnight_L640_30mW_10ms___35',
'240119_SMT_7a_JF646_2.5pm_overnight_L640_30mW_10ms___3',
'240119_SMT_7a_JF646_2.5pm_overnight_L640_30mW_10ms___4',
'240119_SMT_7a_JF646_2.5pm_overnight_L640_30mW_10ms___9',
]

prefixes_7a_2 = [
'240125_rep2_7A_cotransfection_storm_3pm_overnight__11',
'240125_rep2_7A_cotransfection_storm_3pm_overnight__17',
'240125_rep2_7A_cotransfection_storm_3pm_overnight__18',
'240125_rep2_7A_cotransfection_storm_3pm_overnight__23',
'240125_rep2_7A_cotransfection_storm_3pm_overnight__24',
'240125_rep2_7A_cotransfection_storm_3pm_overnight__25',
'240125_rep2_7A_cotransfection_storm_3pm_overnight__28',
'240125_rep2_7A_cotransfection_storm_3pm_overnight__29',
#'240125_rep2_7A_cotransfection_storm_3pm_overnight__2', #drift
'240125_rep2_7A_cotransfection_storm_3pm_overnight__30',
'240125_rep2_7A_cotransfection_storm_3pm_overnight__36',
#'240125_rep2_7A_cotransfection_storm_3pm_overnight__3', #drift 
'240125_rep2_7A_cotransfection_storm_3pm_overnight__40',
'240125_rep2_7A_cotransfection_storm_3pm_overnight__5',
'240125_rep2_7A_cotransfection_storm_3pm_overnight__8',
]

prefixes_7a_3 = [
'240129_STORM_7A_rep3_JF646_3pm_overnight_L640_30mW_10ms__12',
'240129_STORM_7A_rep3_JF646_3pm_overnight_L640_30mW_10ms__16',
'240129_STORM_7A_rep3_JF646_3pm_overnight_L640_30mW_10ms__17',
'240129_STORM_7A_rep3_JF646_3pm_overnight_L640_30mW_10ms__21',
'240129_STORM_7A_rep3_JF646_3pm_overnight_L640_30mW_10ms__23',
'240129_STORM_7A_rep3_JF646_3pm_overnight_L640_30mW_10ms__24',
'240129_STORM_7A_rep3_JF646_3pm_overnight_L640_30mW_10ms__28',
'240129_STORM_7A_rep3_JF646_3pm_overnight_L640_30mW_10ms__2',
'240129_STORM_7A_rep3_JF646_3pm_overnight_L640_30mW_10ms__31',
'240129_STORM_7A_rep3_JF646_3pm_overnight_L640_30mW_10ms__3',
'240129_STORM_7A_rep3_JF646_3pm_overnight_L640_30mW_10ms__6',
'240129_STORM_7A_rep3_JF646_3pm_overnight_L640_30mW_10ms__7',
]


prefixes_7d_1 = [
'240119_SMT_7d_JF646_2.5pm_overnight_L640_30mW_10ms___10',
'240119_SMT_7d_JF646_2.5pm_overnight_L640_30mW_10ms___14',
'240119_SMT_7d_JF646_2.5pm_overnight_L640_30mW_10ms___18',
'240119_SMT_7d_JF646_2.5pm_overnight_L640_30mW_10ms___19',
'240119_SMT_7d_JF646_2.5pm_overnight_L640_30mW_10ms___21',
'240119_SMT_7d_JF646_2.5pm_overnight_L640_30mW_10ms___24',
'240119_SMT_7d_JF646_2.5pm_overnight_L640_30mW_10ms___27',
'240119_SMT_7d_JF646_2.5pm_overnight_L640_30mW_10ms___30',
'240119_SMT_7d_JF646_2.5pm_overnight_L640_30mW_10ms___4',
'240119_SMT_7d_JF646_2.5pm_overnight_L640_30mW_10ms___5',
'240119_SMT_7d_JF646_2.5pm_overnight_L640_30mW_10ms___7',
'240119_SMT_7d_JF646_2.5pm_overnight_L640_30mW_10ms___8'
]

prefixes_7d_2 = [
#'240125_rep2_7D_cotransfection_storm_3pm_overnight__14', #drift
#'240125_rep2_7D_cotransfection_storm_3pm_overnight__15', #strange
#'240125_rep2_7D_cotransfection_storm_3pm_overnight__18', #strange
'240125_rep2_7D_cotransfection_storm_3pm_overnight__21',
'240125_rep2_7D_cotransfection_storm_3pm_overnight__34',
'240125_rep2_7D_cotransfection_storm_3pm_overnight__4',
'240125_rep2_7D_cotransfection_storm_3pm_overnight__7',
'240125_rep2_7D_cotransfection_storm_3pm_overnight__9',
]

prefixes_7d_3 = [
'240129_STORM_7D_rep3_JF646_3pm_overnight_L640_30mW_10ms__11',
'240129_STORM_7D_rep3_JF646_3pm_overnight_L640_30mW_10ms__16',
'240129_STORM_7D_rep3_JF646_3pm_overnight_L640_30mW_10ms__18',
'240129_STORM_7D_rep3_JF646_3pm_overnight_L640_30mW_10ms__19',
'240129_STORM_7D_rep3_JF646_3pm_overnight_L640_30mW_10ms__20',
'240129_STORM_7D_rep3_JF646_3pm_overnight_L640_30mW_10ms__22',
'240129_STORM_7D_rep3_JF646_3pm_overnight_L640_30mW_10ms__23',
#'240129_STORM_7D_rep3_JF646_3pm_overnight_L640_30mW_10ms__2', #drift
#'240129_STORM_7D_rep3_JF646_3pm_overnight_L640_30mW_10ms__31', #bad clustering
'240129_STORM_7D_rep3_JF646_3pm_overnight_L640_30mW_10ms__32',
'240129_STORM_7D_rep3_JF646_3pm_overnight_L640_30mW_10ms__5',
'240129_STORM_7D_rep3_JF646_3pm_overnight_L640_30mW_10ms__9',
]


prefixes_bd_1 = [
'240109_BD_JF646_3pm_overnight_30mW_10ms___19',
'240109_BD_JF646_3pm_overnight_30mW_10ms___22',
#'240109_BD_JF646_3pm_overnight_30mW_10ms___24', #bad clustering
'240109_BD_JF646_3pm_overnight_30mW_10ms___26',
'240109_BD_JF646_3pm_overnight_30mW_10ms___2',
'240109_BD_JF646_3pm_overnight_30mW_10ms___31',
'240109_BD_JF646_3pm_overnight_30mW_10ms___36',
'240109_BD_JF646_3pm_overnight_30mW_10ms___37',
#'240109_BD_JF646_3pm_overnight_30mW_10ms___38', #bad clustering
'240109_BD_JF646_3pm_overnight_30mW_10ms___3',
'240109_BD_JF646_3pm_overnight_30mW_10ms___40',
'240109_BD_JF646_3pm_overnight_30mW_10ms___42',
'240109_BD_JF646_3pm_overnight_30mW_10ms___43',
'240109_BD_JF646_3pm_overnight_30mW_10ms___44',
'240109_BD_JF646_3pm_overnight_30mW_10ms___7',
]

prefixes_bd_2 = [
'240129_STORM_BD_rep2_JF646_3pm_overnight_L640_30mW_10ms__10',
'240129_STORM_BD_rep2_JF646_3pm_overnight_L640_30mW_10ms__12',
'240129_STORM_BD_rep2_JF646_3pm_overnight_L640_30mW_10ms__17',
'240129_STORM_BD_rep2_JF646_3pm_overnight_L640_30mW_10ms__19',
'240129_STORM_BD_rep2_JF646_3pm_overnight_L640_30mW_10ms__1',
'240129_STORM_BD_rep2_JF646_3pm_overnight_L640_30mW_10ms__20',
'240129_STORM_BD_rep2_JF646_3pm_overnight_L640_30mW_10ms__5',
'240129_STORM_BD_rep2_JF646_3pm_overnight_L640_30mW_10ms__7',
]

prefixes_wt_1 = [
'240109_wt_JF646_3pm_overnight_30mW_10ms___10',
#'240109_wt_JF646_3pm_overnight_30mW_10ms___11',
'240109_wt_JF646_3pm_overnight_30mW_10ms___13',
'240109_wt_JF646_3pm_overnight_30mW_10ms___14',
'240109_wt_JF646_3pm_overnight_30mW_10ms___15',
'240109_wt_JF646_3pm_overnight_30mW_10ms___17',
'240109_wt_JF646_3pm_overnight_30mW_10ms___20',
#'240109_wt_JF646_3pm_overnight_30mW_10ms___23', #aberrant
'240109_wt_JF646_3pm_overnight_30mW_10ms___24',
'240109_wt_JF646_3pm_overnight_30mW_10ms___28',
'240109_wt_JF646_3pm_overnight_30mW_10ms___29',
#'240109_wt_JF646_3pm_overnight_30mW_10ms___2', #drift
'240109_wt_JF646_3pm_overnight_30mW_10ms___31',
'240109_wt_JF646_3pm_overnight_30mW_10ms___34',
'240109_wt_JF646_3pm_overnight_30mW_10ms___35',
'240109_wt_JF646_3pm_overnight_30mW_10ms___38',
'240109_wt_JF646_3pm_overnight_30mW_10ms___42',
'240109_wt_JF646_3pm_overnight_30mW_10ms___44',
'240109_wt_JF646_3pm_overnight_30mW_10ms___6',
'240109_wt_JF646_3pm_overnight_30mW_10ms___7',
'240109_wt_JF646_3pm_overnight_30mW_10ms___9',
]

prefixes_wt_2 = [
'240129_STORM_WT_rep2_JF646_3pm_overnight_L640_30mW_10ms__13',
'240129_STORM_WT_rep2_JF646_3pm_overnight_L640_30mW_10ms__14',
'240129_STORM_WT_rep2_JF646_3pm_overnight_L640_30mW_10ms__15',
'240129_STORM_WT_rep2_JF646_3pm_overnight_L640_30mW_10ms__17',
'240129_STORM_WT_rep2_JF646_3pm_overnight_L640_30mW_10ms__19',
'240129_STORM_WT_rep2_JF646_3pm_overnight_L640_30mW_10ms__20',
'240129_STORM_WT_rep2_JF646_3pm_overnight_L640_30mW_10ms__21',
'240129_STORM_WT_rep2_JF646_3pm_overnight_L640_30mW_10ms__23',
'240129_STORM_WT_rep2_JF646_3pm_overnight_L640_30mW_10ms__25',
'240129_STORM_WT_rep2_JF646_3pm_overnight_L640_30mW_10ms__26',
'240129_STORM_WT_rep2_JF646_3pm_overnight_L640_30mW_10ms__27',
'240129_STORM_WT_rep2_JF646_3pm_overnight_L640_30mW_10ms__4',
'240129_STORM_WT_rep2_JF646_3pm_overnight_L640_30mW_10ms__5',
'240129_STORM_WT_rep2_JF646_3pm_overnight_L640_30mW_10ms__7',
'240129_STORM_WT_rep2_JF646_3pm_overnight_L640_30mW_10ms__8',
]

prefixes_si_1 = [
'240117_STORM_LIVE_silenceBRD4_646_3.5pm_overnight_L640_30mW_10ms___10',
'240117_STORM_LIVE_silenceBRD4_646_3.5pm_overnight_L640_30mW_10ms___11',
'240117_STORM_LIVE_silenceBRD4_646_3.5pm_overnight_L640_30mW_10ms___12',
'240117_STORM_LIVE_silenceBRD4_646_3.5pm_overnight_L640_30mW_10ms___14',
'240117_STORM_LIVE_silenceBRD4_646_3.5pm_overnight_L640_30mW_10ms___15',
'240117_STORM_LIVE_silenceBRD4_646_3.5pm_overnight_L640_30mW_10ms___16',
'240117_STORM_LIVE_silenceBRD4_646_3.5pm_overnight_L640_30mW_10ms___29',
'240117_STORM_LIVE_silenceBRD4_646_3.5pm_overnight_L640_30mW_10ms___4',
'240117_STORM_LIVE_silenceBRD4_646_3.5pm_overnight_L640_30mW_10ms___5',
'240117_STORM_LIVE_silenceBRD4_646_3.5pm_overnight_L640_30mW_10ms___6'
]

prefixes_ctrl_1 = [
'240114_Live_Control_JF646_3pm_overnight_L640_30mW_10ms__10',
'240114_Live_Control_JF646_3pm_overnight_L640_30mW_10ms__16',
'240114_Live_Control_JF646_3pm_overnight_L640_30mW_10ms__19',
'240114_Live_Control_JF646_3pm_overnight_L640_30mW_10ms__20',
'240114_Live_Control_JF646_3pm_overnight_L640_30mW_10ms__21',
'240114_Live_Control_JF646_3pm_overnight_L640_30mW_10ms__22',
'240114_Live_Control_JF646_3pm_overnight_L640_30mW_10ms__5'
]

prefixes_7a_2 = []
prefixes_7a_3 = []
prefixes_7d_2 = []
prefixes_7d_3 = []
prefixes_wt_2 = []

with open('run_cluster.json', 'r') as f:
    config = json.load(f)
    
plot_7a=True
plot_7d=True
plot_bd=True
plot_wt=True
plot_si=False
plot_ctrl=False

show_clusters=False
num_samples=20000
fig,ax=plt.subplots(figsize=(4,4))
bins = np.logspace(3,5,50)

if plot_si:
    grads_si = []
    for prefix in prefixes_si_1:
        print("Processing " + prefix)
        spots = pd.read_csv(config['path5']+prefix+'.csv')
        #kde= imread(config['path2']+prefix+'-render.tif')
        spots = spots.loc[spots['uncertainty [nm]'] < 20]
        #plt.imshow(kde,cmap='gray',vmin=0,vmax=0.5)
        try:
            spots = spots.sample(num_samples)
            clust = RTCluster(spots,config['r'],config['T'])
            scores,H = clust.cluster(showK=False,show_clusters=show_clusters,plot_ind_fit=False,
                               fit_model=False,show_fit=False,hw=config['hw'])
            grads_si += H
        except:
            pass
 
    vals6, bins6 = np.histogram(grads_si,bins=bins,density=True)
    ax.scatter(bins6[:-1],vals6,color='lime',alpha=0.5,label='siBRD4')
            
if plot_ctrl:
    grads_ctrl = []
    for prefix in prefixes_ctrl_1:
        print("Processing " + prefix)
        spots = pd.read_csv(config['path5']+prefix+'.csv')
        #kde= imread(config['path2']+prefix+'-render.tif')
        spots = spots.loc[spots['uncertainty [nm]'] < 20]
        #plt.imshow(kde,cmap='gray',vmin=0,vmax=0.5)
        try:
            spots = spots.sample(num_samples)
            clust = RTCluster(spots,config['r'],config['T'])
            scores,H = clust.cluster(showK=False,show_clusters=show_clusters,plot_ind_fit=False,
                               fit_model=False,show_fit=False,hw=config['hw'])
            grads_ctrl += H
        except:
            pass

    vals5, bins5 = np.histogram(grads_ctrl,bins=bins,density=True)
    ax.scatter(bins5[:-1],vals5,color='purple',alpha=0.5,label='Control')

if plot_7a:
    grads_7a = []
    for prefix in prefixes_7a_1:
        print("Processing " + prefix)
        spots = pd.read_csv(config['path2']+prefix+'.csv')
        #kde= imread(config['path2']+prefix+'-render.tif')
        spots = spots.loc[spots['uncertainty [nm]'] < 20]
        #plt.imshow(kde,cmap='gray',vmin=0,vmax=0.5)
        try:
            spots = spots.sample(num_samples)
            clust = RTCluster(spots,config['r'],config['T'])
            scores,H = clust.cluster(showK=False,show_clusters=show_clusters,plot_ind_fit=False,
                               fit_model=False,show_fit=False,hw=config['hw'])
            grads_7a += H
        except:
            pass
                
    for prefix in prefixes_7a_2:
        print("Processing " + prefix)
        spots = pd.read_csv(config['path3']+prefix+'.csv')
        #kde= imread(config['path3']+prefix+'-render.tif')
        spots = spots.loc[spots['uncertainty [nm]'] < 20]
        #plt.imshow(kde,cmap='gray',vmin=0,vmax=0.5)
        try:
            spots = spots.sample(num_samples)
            clust = RTCluster(spots,config['r'],config['T'])
            scores,H = clust.cluster(showK=False,show_clusters=show_clusters,plot_ind_fit=False,
                               fit_model=False,show_fit=False,hw=config['hw'])
            grads_7a += H
        except:
            pass
        
    for prefix in prefixes_7a_3:
        print("Processing " + prefix)
        spots = pd.read_csv(config['path4']+prefix+'.csv')
        #kde= imread(config['path4']+prefix+'-render.tif')
        spots = spots.loc[spots['uncertainty [nm]'] < 20]
        #plt.imshow(kde,cmap='gray',vmin=0,vmax=0.5)
        try:
            spots = spots.sample(num_samples)
            clust = RTCluster(spots,config['r'],config['T'])
            scores,H = clust.cluster(showK=False,show_clusters=show_clusters,plot_ind_fit=False,
                               fit_model=False,show_fit=False,hw=config['hw'])
            grads_7a += H
        except:
            pass

    vals1, bins1 = np.histogram(grads_7a,bins=bins,density=True)
    ax.scatter(bins1[:-1],vals1,color='blue',alpha=0.5,label='7A')


if plot_7d:
    grads_7d = []
    for prefix in prefixes_7d_1:
        print("Processing " + prefix)
        spots = pd.read_csv(config['path2']+prefix+'.csv')
        #kde= imread(config['path2']+prefix+'-render.tif')
        spots = spots.loc[spots['uncertainty [nm]'] < 20]
        #plt.imshow(kde,cmap='gray',vmin=0,vmax=0.5)
        try:
            spots = spots.sample(num_samples)
            clust = RTCluster(spots,config['r'],config['T'])
            scores,H = clust.cluster(showK=False,show_clusters=show_clusters,plot_ind_fit=False,
                               fit_model=False,show_fit=False,hw=config['hw'])
            grads_7d += H
        except:
            pass
            
    for prefix in prefixes_7d_2:
        print("Processing " + prefix)
        spots = pd.read_csv(config['path3']+prefix+'.csv')
        #kde= imread(config['path3']+prefix+'-render.tif')
        spots = spots.loc[spots['uncertainty [nm]'] < 20]
        #plt.imshow(kde,cmap='gray',vmin=0,vmax=0.5)
        try:
            spots = spots.sample(num_samples)
            clust = RTCluster(spots,config['r'],config['T'])
            scores,H = clust.cluster(showK=False,show_clusters=show_clusters,plot_ind_fit=False,
                               fit_model=False,show_fit=False,hw=config['hw'])
            grads_7d += H
        except:
            pass
        
    for prefix in prefixes_7d_3:
        print("Processing " + prefix)
        spots = pd.read_csv(config['path4']+prefix+'.csv')
        #kde= imread(config['path4']+prefix+'-render.tif')
        spots = spots.loc[spots['uncertainty [nm]'] < 20]
        #plt.imshow(kde,cmap='gray',vmin=0,vmax=0.5)
        try:
            spots = spots.sample(num_samples)
            clust = RTCluster(spots,config['r'],config['T'])
            scores,H = clust.cluster(showK=False,show_clusters=show_clusters,plot_ind_fit=False,
                               fit_model=False,show_fit=False,hw=config['hw'])
            grads_7d += H
        except:
            pass

    vals2, bins2 = np.histogram(grads_7d,bins=bins,density=True)
    ax.scatter(bins2[:-1],vals2,color='cyan',alpha=0.5,label='7D')
    
if plot_bd:
    grads_bd = []
    for prefix in prefixes_bd_1:
        print("Processing " + prefix)
        spots = pd.read_csv(config['path1']+prefix+'.csv')
        #kde= imread(config['path1']+prefix+'-render.tif')
        spots = spots.loc[spots['uncertainty [nm]'] < 20]
        #plt.imshow(kde,cmap='gray',vmin=0,vmax=0.5)
        try:
            spots = spots.sample(num_samples)
            clust = RTCluster(spots,config['r'],config['T'])
            scores,H = clust.cluster(showK=False,show_clusters=show_clusters,plot_ind_fit=False,
                               fit_model=False,show_fit=False,hw=config['hw'])
            grads_bd += H
        except:
            pass
        
    for prefix in prefixes_bd_2:
        print("Processing " + prefix)
        spots = pd.read_csv(config['path4']+prefix+'.csv')
        #kde= imread(config['path4']+prefix+'-render.tif')
        spots = spots.loc[spots['uncertainty [nm]'] < 20]
        #plt.imshow(kde,cmap='gray',vmin=0,vmax=0.5)
        try:
            spots = spots.sample(num_samples)
            clust = RTCluster(spots,config['r'],config['T'])
            scores,H = clust.cluster(showK=False,show_clusters=show_clusters,plot_ind_fit=False,
                               fit_model=False,show_fit=False,hw=config['hw'])
            grads_bd += H
        except:
            pass
            
    vals3, bins3 = np.histogram(grads_bd,bins=bins,density=True)
    ax.scatter(bins3[:-1],vals3,color='red',alpha=0.5,label='BD')
        
if plot_wt:
    grads_wt = []
    for prefix in prefixes_wt_1:
        print("Processing " + prefix)
        spots = pd.read_csv(config['path1']+prefix+'.csv')
        #kde= imread(config['path1']+prefix+'-render.tif')
        spots = spots.loc[spots['uncertainty [nm]'] < 20]
        #plt.imshow(kde,cmap='gray',vmin=0,vmax=0.5)
        try:
            spots = spots.sample(num_samples)
            clust = RTCluster(spots,config['r'],config['T'])
            scores,H = clust.cluster(showK=False,show_clusters=show_clusters,plot_ind_fit=False,
                               fit_model=False,show_fit=False,hw=config['hw'])
            grads_wt += H
        except:
            pass
        
    for prefix in prefixes_wt_2:
        print("Processing " + prefix)
        spots = pd.read_csv(config['path4']+prefix+'.csv')
        #kde= imread(config['path4']+prefix+'-render.tif')
        spots = spots.loc[spots['uncertainty [nm]'] < 20]
        #plt.imshow(kde,cmap='gray',vmin=0,vmax=0.5)
        try:
            spots = spots.sample(num_samples)
            clust = RTCluster(spots,config['r'],config['T'])
            scores,H = clust.cluster(showK=False,show_clusters=show_clusters,plot_ind_fit=False,
                               fit_model=False,show_fit=False,hw=config['hw'])
            grads_wt += H
        except:
            pass
           
    vals4, bins4 = np.histogram(grads_wt,bins=bins,density=True)
    ax.scatter(bins4[:-1],vals4,color='black',alpha=0.5,label='WT')      


ax.set_xlabel(r'$R_{g}^{2} (\mathrm{nm}^{2})$',fontsize=12)
ax.set_ylabel('Density',fontsize=12)
ax.set_xscale('log')
ax.set_yscale('log')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend()
plt.tight_layout()
plt.show()                        
