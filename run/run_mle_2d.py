from pipes import PipelineMLE2D
from DeepSMLM.torch.dataset import SMLMDataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json


prefixes = [
'231206_Control_646_2pm_1hour_L640_5mW_100ms__5'
]

with open('run_mle_2d.json', 'r') as f:
    config = json.load(f)

for prefix in prefixes:
    print("Processing " + prefix)
    dataset = SMLMDataset(config['datapath']+prefix,prefix)
    pipe = PipelineMLE2D(config,dataset)
    pipe.localize(plot_spots=False,plot_fit=False,fit=True,tmax=5)
