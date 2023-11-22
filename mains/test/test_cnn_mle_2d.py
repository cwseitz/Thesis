import numpy as np
from tests import *
import json


config = {
"modelpath": "/home/cwseitz/Desktop/Torch/Models2D/",
"gain": "cmos/gain.npz",
"offset": "cmos/offset.npz",
"var": "cmos/var.npz",
"sigma": 0.92,
"N0": 1000,
"B0": 0,
"texp": 1.0,
"eta": 0.8,
"particles" : 5,
"modelname": "1121_211821",
"pixel_size": 108.3,
"min_sigma": 1,
"max_sigma": 3,
"num_sigma": 5,
"threshold": 5.0,
"overlap": 0.5,
"lr": [1e-8,1e-8,0.0,0.0],
"patchw": 3,
"ring_radius":1
}

cnn2dtest = CNN2D_MLE2D_Ring_Test(config)
cnn2dtest.test()



