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
    pipe.localize(plot_spots=False,plot_fit=False,fit=True,tmax=2000)
    spots = pd.read_csv(config['analpath'] + prefix + '/' + prefix + '_spots.csv')
    spots = spots.loc[spots['conv'] == True]
    spots = spots.loc[(spots['x_err']  < 0.1) & (spots['y_err']  < 0.1)]    
    render = KDE(spots).forward(sigma=2.0)
    imsave(config['analpath']+prefix+'/'+prefix+'-kde.tif',render)
    #make_animation(dataset.stack,spots)

