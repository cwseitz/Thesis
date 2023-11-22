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
"modelname": "1118_185147",
"pixel_size_lateral": 108.3,
"thresh_cnn": 30,
"radius": 4
}

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
"modelname": "1121_204931",
"pixel_size_lateral": 108.3,
"min_sigma": 1,
"max_sigma": 3,
"num_sigma": 5,
"threshold": 5.0,
"overlap": 0.5
}

cnn2dtest = CNN2D_Test(config)
cnn2dtest.test()



