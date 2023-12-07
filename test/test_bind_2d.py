import numpy as np
from tests import *
import json

config = {
"texp": 1.0,
"eta": 0.8,
"gain": "cmos/gain.npz",
"offset": "cmos/offset.npz",
"var": "cmos/var.npz",
"sigma": 0.92,
"N0": 1000,
"B0": 0,
"patchw": 3,
"particles":5,
"ring_radius": 1,
"pixel_size": 108.3,
"modelpath": "/home/cwseitz/Desktop/Torch/Models2D/",
"modelname": "1123_164800",
"pixel_size_lateral": 108.3,
"min_sigma": 1,
"max_sigma": 3,
"num_sigma": 5,
"threshold": 5.0,
"overlap": 0.5,
"wprior": 5.0
}

map2dtest = BIND_2DMix_Test(config)
map2dtest.test()




