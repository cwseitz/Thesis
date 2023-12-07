import numpy as np
from tests import *
import json

config = {
"modelpath": "/home/cwseitz/Desktop/Torch/Models2D/",
"nx": 20,
"ny": 20,
"sigma": 0.92,
"N0": 1000,
"B0": 0,
"texp": 1.0,
"eta": 0.8,
"radius": 5.0,
"nspots" : 5,
"modelname": "1204_230202",
"pixel_size_lateral": 108.3,
"threshold": 1.0,
"overlap": 0.5,
"min_sigma":1,
"max_sigma":3,
"num_sigma":5
}

cnn2dtest = CNNDouble_Test(config)
cnn2dtest.test()



