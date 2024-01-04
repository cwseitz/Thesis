import json
from pipes import *
import pandas as pd
import trackpy as tp
from DeepSMLM.torch.dataset import SMLMDataset
from BaseSMLM.utils import make_animation, Tracker2D
from skimage.io import imread

prefixes = [
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

prefixes = [
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

prefixes = [
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

prefixes = [
'231206_Control_646_2pm_1hour_L640_5mW_100ms__1',
'231206_Control_646_2pm_1hour_L640_5mW_100ms__10',
'231206_Control_646_2pm_1hour_L640_5mW_100ms__14',
'231206_Control_646_2pm_1hour_L640_5mW_100ms__15',
'231206_Control_646_2pm_1hour_L640_5mW_100ms__17',
'231206_Control_646_2pm_1hour_L640_5mW_100ms__19',
'231206_Control_646_2pm_1hour_L640_5mW_100ms__24',
'231206_Control_646_2pm_1hour_L640_5mW_100ms__3',
'231206_Control_646_2pm_1hour_L640_5mW_100ms__35',
'231206_Control_646_2pm_1hour_L640_5mW_100ms__4',
'231206_Control_646_2pm_1hour_L640_5mW_100ms__5',
'231206_Control_646_2pm_1hour_L640_5mW_100ms__6',
'231206_Control_646_2pm_1hour_L640_5mW_100ms__8',
'231206_Control_646_2pm_1hour_L640_5mW_100ms__9',
]

with open('run_track_2d.json', 'r') as f:
    config = json.load(f)

for n,prefix in enumerate(prefixes):
    print("Processing " + prefix)
    dataset = SMLMDataset(config['datapath']+prefix,prefix)
    tracker = Tracker2D()
    spots = pd.read_csv(config['analpath'] + prefix + '/' + prefix + '_spots.csv')
    spots = spots.loc[spots['conv'] == True]
    spots = spots.loc[(spots['x_err']  < 0.03) & (spots['y_err']  < 0.03)]
    linked = tracker.link(spots)
    linked = tp.filter_stubs(linked,80)
    tp.plot_traj(linked,superimpose=dataset.stack[0],
                 pos_columns=['y_mle','x_mle'])
                 
    """
    linked.to_csv(config['analpath'] + prefix + '/' + prefix + '_link.csv')

    im = tracker.imsd(linked)
    im.to_csv(config['analpath'] + prefix + '/' + prefix + '_msd.csv')
    vh = tracker.vanhove(linked)
    vh.to_csv(config['analpath'] + prefix + '/' + prefix + '_vhe.csv')
    """



