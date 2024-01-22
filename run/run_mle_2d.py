from pipes import PipelineMLE2D
from DeepSMLM.torch.dataset import SMLMDataset
from BaseSMLM.utils import KDE, make_animation
from skimage.io import imsave
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

prefixes = [
'231222_BD_5mw_100ms_JF646_2+2pm_overnight_30mW_10ms__11',
'231222_BD_5mw_100ms_JF646_2+2pm_overnight_30mW_10ms__12',
'231222_BD_5mw_100ms_JF646_2+2pm_overnight_30mW_10ms__13',
'231222_BD_5mw_100ms_JF646_2+2pm_overnight_30mW_10ms__14',
'231222_BD_5mw_100ms_JF646_2+2pm_overnight_30mW_10ms__15',
'231222_BD_5mw_100ms_JF646_2+2pm_overnight_30mW_10ms__16',
'231222_BD_5mw_100ms_JF646_2+2pm_overnight_30mW_10ms__17',
'231222_BD_5mw_100ms_JF646_2+2pm_overnight_30mW_10ms__21',
'231222_BD_5mw_100ms_JF646_2+2pm_overnight_30mW_10ms__25',
'231222_BD_5mw_100ms_JF646_2+2pm_overnight_30mW_10ms__26',
'231222_BD_5mw_100ms_JF646_2+2pm_overnight_30mW_10ms__27',
'231222_BD_5mw_100ms_JF646_2+2pm_overnight_30mW_10ms__8',
'231225_BD_5mw_100ms_JF646_4pm_overnight_30mw_10ms__11',
'231225_BD_5mw_100ms_JF646_4pm_overnight_30mw_10ms__4',
'231225_BD_5mw_100ms_JF646_4pm_overnight_30mw_10ms__5'
]

prefixes = [
'231112_JQ1_8hours_JF646_7pm_overnight_30mW_10ms_50'
]

prefixes = [
'231226_BD_5mw_100ms_JF646_2pm_1hour__14',
'231226_BD_5mw_100ms_JF646_2pm_1hour__18',
'231226_BD_5mw_100ms_JF646_2pm_1hour__20',
'231226_BD_5mw_100ms_JF646_2pm_1hour__21',
'231226_BD_5mw_100ms_JF646_2pm_1hour__33',
'231226_BD_5mw_100ms_JF646_2pm_1hour__34',
'231226_BD_5mw_100ms_JF646_2pm_1hour__35',
'231226_BD_5mw_100ms_JF646_2pm_1hour__3',
'231226_BD_5mw_100ms_JF646_2pm_1hour__42',
'231226_BD_5mw_100ms_JF646_2pm_1hour__5',
'231226_BD_5mw_100ms_JF646_2pm_1hour__9'
]

prefixes = [
'231028_CONTROL_H2B 1000ng_7pm_15h__L640-20mW_10ms__4',
'231028_CONTROL_H2B 1000ng_7pm_15h__L640-30mW_10ms__1',
'231028_CONTROL_H2B 1000ng_7pm_15h__L640-30mW_10ms__22',
'231028_CONTROL_H2B 1000ng_7pm_15h__L640-30mW_10ms__23',
'231028_CONTROL_H2B 1000ng_7pm_15h__L640-30mW_10ms__2',
'231028_CONTROL_H2B 1000ng_7pm_15h__L640-30mW_10ms__31',
'231028_CONTROL_H2B 1000ng_7pm_15h__L640-30mW_10ms__32',
'231028_CONTROL_H2B 1000ng_7pm_15h__L640-30mW_10ms__7',
'231028_JQ1_1uM_8hours_H2B 1000ng_7pm_15h__L640-30mW_10ms__11',
'231028_JQ1_1uM_8hours_H2B 1000ng_7pm_15h__L640-30mW_10ms__1',
'231028_JQ1_1uM_8hours_H2B 1000ng_7pm_15h__L640-30mW_10ms__27',
'231028_JQ1_1uM_8hours_H2B 1000ng_7pm_15h__L640-30mW_10ms__28',
'231028_JQ1_1uM_8hours_H2B 1000ng_7pm_15h__L640-30mW_10ms__2',
'231028_JQ1_1uM_8hours_H2B 1000ng_7pm_15h__L640-30mW_10ms__31',
'231028_JQ1_1uM_8hours_H2B 1000ng_7pm_15h__L640-30mW_10ms__32',
'231028_JQ1_1uM_8hours_H2B 1000ng_7pm_15h__L640-30mW_10ms__33',
'231028_JQ1_1uM_8hours_H2B 1000ng_7pm_15h__L640-30mW_10ms__6'
]

prefixes = [
#'231112_JQ1_8hours_JF646_7pm_overnight_30mW_10ms_10',
'231112_JQ1_8hours_JF646_7pm_overnight_30mW_10ms_30',
'231112_JQ1_8hours_JF646_7pm_overnight_30mW_10ms_36',
'231112_JQ1_8hours_JF646_7pm_overnight_30mW_10ms_38',
'231112_JQ1_8hours_JF646_7pm_overnight_30mW_10ms_39',
'231112_JQ1_8hours_JF646_7pm_overnight_30mW_10ms_3',
'231112_JQ1_8hours_JF646_7pm_overnight_30mW_10ms_41',
'231112_JQ1_8hours_JF646_7pm_overnight_30mW_10ms_47',
'231112_JQ1_8hours_JF646_7pm_overnight_30mW_10ms_4',
'231112_JQ1_8hours_JF646_7pm_overnight_30mW_10ms_50',
'231112_JQ1_8hours_JF646_7pm_overnight_30mW_10ms_52',
'231112_JQ1_8hours_JF646_7pm_overnight_30mW_10ms_5',
'231112_JQ1_8hours_JF646_7pm_overnight_30mW_10ms_7'
]

prefixes = [
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

prefixes = [
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

prefixes = [
'231226_WT_5mw_100ms_JF646_2pm_1hour__15',
'231226_WT_5mw_100ms_JF646_2pm_1hour__16',
'231226_WT_5mw_100ms_JF646_2pm_1hour__23',
'231226_WT_5mw_100ms_JF646_2pm_1hour__24',
'231226_WT_5mw_100ms_JF646_2pm_1hour__26',
'231226_WT_5mw_100ms_JF646_2pm_1hour__33',
'231226_WT_5mw_100ms_JF646_2pm_1hour__3'
]

prefixes = [
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

prefixes = [
'240108_Control_646_2pm_1hour_L640_5mW_100ms___11',
'240108_Control_646_2pm_1hour_L640_5mW_100ms___13',
'240108_Control_646_2pm_1hour_L640_5mW_100ms___17',
'240108_Control_646_2pm_1hour_L640_5mW_100ms___21',
'240108_Control_646_2pm_1hour_L640_5mW_100ms___27',
'240108_Control_646_2pm_1hour_L640_5mW_100ms___7',
'240108_Control_646_2pm_1hour_L640_5mW_100ms___9'
]

prefixes = [
'240108_jq1_646_2pm_1hour_L640_5mW_100ms___15',
'240108_jq1_646_2pm_1hour_L640_5mW_100ms___16',
'240108_jq1_646_2pm_1hour_L640_5mW_100ms___17',
'240108_jq1_646_2pm_1hour_L640_5mW_100ms___24',
'240108_jq1_646_2pm_1hour_L640_5mW_100ms___25',
'240108_jq1_646_2pm_1hour_L640_5mW_100ms___27',
'240108_jq1_646_2pm_1hour_L640_5mW_100ms___35',
'240108_jq1_646_2pm_1hour_L640_5mW_100ms___4',
'240108_jq1_646_2pm_1hour_L640_5mW_100ms___6'
]

prefixes = [
'240109_BD_JF646_3pm_overnight_30mW_10ms___19',
'240109_BD_JF646_3pm_overnight_30mW_10ms___22',
'240109_BD_JF646_3pm_overnight_30mW_10ms___24',
'240109_BD_JF646_3pm_overnight_30mW_10ms___26',
'240109_BD_JF646_3pm_overnight_30mW_10ms___2',
'240109_BD_JF646_3pm_overnight_30mW_10ms___31',
'240109_BD_JF646_3pm_overnight_30mW_10ms___36',
'240109_BD_JF646_3pm_overnight_30mW_10ms___37',
'240109_BD_JF646_3pm_overnight_30mW_10ms___38',
'240109_BD_JF646_3pm_overnight_30mW_10ms___3',
'240109_BD_JF646_3pm_overnight_30mW_10ms___40',
'240109_BD_JF646_3pm_overnight_30mW_10ms___42',
'240109_BD_JF646_3pm_overnight_30mW_10ms___43',
'240109_BD_JF646_3pm_overnight_30mW_10ms___44',
'240109_BD_JF646_3pm_overnight_30mW_10ms___7'
]

prefixes = [
'240109_wt_JF646_3pm_overnight_30mW_10ms___10',
'240109_wt_JF646_3pm_overnight_30mW_10ms___11',
'240109_wt_JF646_3pm_overnight_30mW_10ms___13',
'240109_wt_JF646_3pm_overnight_30mW_10ms___14',
'240109_wt_JF646_3pm_overnight_30mW_10ms___15',
'240109_wt_JF646_3pm_overnight_30mW_10ms___17',
'240109_wt_JF646_3pm_overnight_30mW_10ms___20',
'240109_wt_JF646_3pm_overnight_30mW_10ms___23',
'240109_wt_JF646_3pm_overnight_30mW_10ms___24',
'240109_wt_JF646_3pm_overnight_30mW_10ms___28',
'240109_wt_JF646_3pm_overnight_30mW_10ms___29',
'240109_wt_JF646_3pm_overnight_30mW_10ms___2',
'240109_wt_JF646_3pm_overnight_30mW_10ms___31',
'240109_wt_JF646_3pm_overnight_30mW_10ms___34',
'240109_wt_JF646_3pm_overnight_30mW_10ms___35',
'240109_wt_JF646_3pm_overnight_30mW_10ms___38',
'240109_wt_JF646_3pm_overnight_30mW_10ms___42',
'240109_wt_JF646_3pm_overnight_30mW_10ms___44',
'240109_wt_JF646_3pm_overnight_30mW_10ms___6',
'240109_wt_JF646_3pm_overnight_30mW_10ms___7',
'240109_wt_JF646_3pm_overnight_30mW_10ms___9'
]

prefixes = [
'231213_Si BRD4_JF646_2pm_overnight_L640_30mW_10ms___14',
'231213_Si BRD4_JF646_2pm_overnight_L640_30mW_10ms___15',
'231213_Si BRD4_JF646_2pm_overnight_L640_30mW_10ms___17',
'231213_Si BRD4_JF646_2pm_overnight_L640_30mW_10ms___18',
'231213_Si BRD4_JF646_2pm_overnight_L640_30mW_10ms___19',
'231213_Si BRD4_JF646_2pm_overnight_L640_30mW_10ms___20',
'231213_Si BRD4_JF646_2pm_overnight_L640_30mW_10ms___22',
'231213_Si BRD4_JF646_2pm_overnight_L640_30mW_10ms___25',
'231213_Si BRD4_JF646_2pm_overnight_L640_30mW_10ms___26',
'231213_Si BRD4_JF646_2pm_overnight_L640_30mW_10ms___27',
'231213_Si BRD4_JF646_2pm_overnight_L640_30mW_10ms___2',
'231213_Si BRD4_JF646_2pm_overnight_L640_30mW_10ms___32',
'231213_Si BRD4_JF646_2pm_overnight_L640_30mW_10ms___5',
'231213_Si BRD4_JF646_2pm_overnight_L640_30mW_10ms___7',
'231213_Si BRD4_JF646_2pm_overnight_L640_30mW_10ms___8'
]

prefixes = [
'240114_JQ1_4hours directly_646_2pm_1hour_L640_5mW_100ms___10',
'240114_JQ1_4hours directly_646_2pm_1hour_L640_5mW_100ms___16',
'240114_JQ1_4hours directly_646_2pm_1hour_L640_5mW_100ms___19',
'240114_JQ1_4hours directly_646_2pm_1hour_L640_5mW_100ms___22',
'240114_JQ1_4hours directly_646_2pm_1hour_L640_5mW_100ms___25',
'240114_JQ1_4hours directly_646_2pm_1hour_L640_5mW_100ms___28',
'240114_JQ1_4hours directly_646_2pm_1hour_L640_5mW_100ms___2',
'240114_JQ1_4hours directly_646_2pm_1hour_L640_5mW_100ms___30'
]

prefixes = [
'240114_JQ1_8hours directly_646_2pm_1hour_L640_5mW_100ms___11',
'240114_JQ1_8hours directly_646_2pm_1hour_L640_5mW_100ms___13',
'240114_JQ1_8hours directly_646_2pm_1hour_L640_5mW_100ms___15',
'240114_JQ1_8hours directly_646_2pm_1hour_L640_5mW_100ms___16',
'240114_JQ1_8hours directly_646_2pm_1hour_L640_5mW_100ms___18',
'240114_JQ1_8hours directly_646_2pm_1hour_L640_5mW_100ms___19',
'240114_JQ1_8hours directly_646_2pm_1hour_L640_5mW_100ms___27',
'240114_JQ1_8hours directly_646_2pm_1hour_L640_5mW_100ms___3',
'240114_JQ1_8hours directly_646_2pm_1hour_L640_5mW_100ms___5'
]

prefixes = [
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

prefixes = [
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

prefixes = [
'240114_Live_Control_JF646_3pm_overnight_L640_30mW_10ms__10',
'240114_Live_Control_JF646_3pm_overnight_L640_30mW_10ms__16',
'240114_Live_Control_JF646_3pm_overnight_L640_30mW_10ms__19',
'240114_Live_Control_JF646_3pm_overnight_L640_30mW_10ms__20',
'240114_Live_Control_JF646_3pm_overnight_L640_30mW_10ms__21',
'240114_Live_Control_JF646_3pm_overnight_L640_30mW_10ms__22',
'240114_Live_Control_JF646_3pm_overnight_L640_30mW_10ms__5'
]

prefixes = [
'240119_SMT_7a_JF646_2.5pm_overnight_L640_30mW_10ms___11',
'240119_SMT_7a_JF646_2.5pm_overnight_L640_30mW_10ms___16',
'240119_SMT_7a_JF646_2.5pm_overnight_L640_30mW_10ms___19',
'240119_SMT_7a_JF646_2.5pm_overnight_L640_30mW_10ms___1',
'240119_SMT_7a_JF646_2.5pm_overnight_L640_30mW_10ms___20',
'240119_SMT_7a_JF646_2.5pm_overnight_L640_30mW_10ms___21',
'240119_SMT_7a_JF646_2.5pm_overnight_L640_30mW_10ms___22',
'240119_SMT_7a_JF646_2.5pm_overnight_L640_30mW_10ms___2',
'240119_SMT_7a_JF646_2.5pm_overnight_L640_30mW_10ms___3',
'240119_SMT_7a_JF646_2.5pm_overnight_L640_30mW_10ms___9'
]

prefixes = [
'240119_SMT_7d_JF646_2.5pm_overnight_L640_30mW_10ms___10',
'240119_SMT_7d_JF646_2.5pm_overnight_L640_30mW_10ms___14', 
'240119_SMT_7d_JF646_2.5pm_overnight_L640_30mW_10ms___18',
'240119_SMT_7d_JF646_2.5pm_overnight_L640_30mW_10ms___19',
'240119_SMT_7d_JF646_2.5pm_overnight_L640_30mW_10ms___24',
'240119_SMT_7d_JF646_2.5pm_overnight_L640_30mW_10ms___27',
'240119_SMT_7d_JF646_2.5pm_overnight_L640_30mW_10ms___4',
'240119_SMT_7d_JF646_2.5pm_overnight_L640_30mW_10ms___5',
'240119_SMT_7d_JF646_2.5pm_overnight_L640_30mW_10ms___7',
'240119_SMT_7d_JF646_2.5pm_overnight_L640_30mW_10ms___8'
]

prefixes = [
'240119_SMT_7D_JF646_1.5pm_1h_L640_5mW_100ms___12',
'240119_SMT_7D_JF646_1.5pm_1h_L640_5mW_100ms___3',
'240119_SMT_7D_JF646_1.5pm_1h_L640_5mW_100ms___6',
'240121_SMT_7D_JF646_1.5pm_1h_L640_5mW_100m_10',
'240121_SMT_7D_JF646_1.5pm_1h_L640_5mW_100m_15',
'240121_SMT_7D_JF646_1.5pm_1h_L640_5mW_100m_2',
'240121_SMT_7D_JF646_1.5pm_1h_L640_5mW_100m_4',
'240121_SMT_7D_JF646_1.5pm_1h_L640_5mW_100m_5',
'240121_SMT_7D_JF646_1.5pm_1h_L640_5mW_100m_7',
'240121_SMT_7D_JF646_1.5pm_1h_L640_5mW_100m_8'
]

prefixes = [
'240118_SMT_7A_JF646_1.5pm_1h_L640_5mW_100ms___10',
'240118_SMT_7A_JF646_1.5pm_1h_L640_5mW_100ms___14',
'240118_SMT_7A_JF646_1.5pm_1h_L640_5mW_100ms___17',
'240118_SMT_7A_JF646_1.5pm_1h_L640_5mW_100ms___19',
'240118_SMT_7A_JF646_1.5pm_1h_L640_5mW_100ms___1',
'240118_SMT_7A_JF646_1.5pm_1h_L640_5mW_100ms___21',
'240118_SMT_7A_JF646_1.5pm_1h_L640_5mW_100ms___24',
'240118_SMT_7A_JF646_1.5pm_1h_L640_5mW_100ms___29',
'240118_SMT_7A_JF646_1.5pm_1h_L640_5mW_100ms___31',
'240118_SMT_7A_JF646_1.5pm_1h_L640_5mW_100ms___4',
'240118_SMT_7A_JF646_1.5pm_1h_L640_5mW_100ms___7'
]

def cc(image):
    fig,ax=plt.subplots()
    image_product = np.fft.fft2(image) * np.fft.fft2(image).conj()
    cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
    ax.imshow(cc_image.real)
    ax.set_axis_off()
    ax.set_title("Cross-correlation")
    plt.show()

with open('run_mle_2d.json', 'r') as f:
    config = json.load(f)

for prefix in prefixes:
    print("Processing " + prefix)
    dataset = SMLMDataset(config['datapath']+prefix,prefix)
    pipe = PipelineMLE2D(config,dataset)
    pipe.localize(plot_spots=False,plot_fit=False,fit=True,tmax=200)
    #spots = pd.read_csv(config['analpath'] + prefix + '/' + prefix + '_spots.csv')
    #spots = spots.loc[spots['conv'] == True]
    #spots = spots.loc[(spots['x_err']  < 0.1) & (spots['y_err']  < 0.1)]    
    #render = KDE(spots).forward(sigma=2.0)
    #imsave(config['analpath']+prefix+'/'+prefix+'-kde',render)
    #make_animation(dataset.stack,spots)

