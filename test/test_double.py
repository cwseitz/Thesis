import numpy as np
from tests import *
import json

config = {
"modelpath": "/home/cwseitz/Desktop/Torch/Models2D/DeepSTORM/",
"nx": 20,
"ny": 20,
"kwargs":{
"texp": 1.0,
"eta": 0.5,
"sigma": 0.92,
"N0": 10,
"gain": 1.0,
"offset": 0.0,
"var": 0.0,
"nframes":1000
},
"disc_radius": 2.0,
"nspots" : 5,
"modelname": "1206_150420",
"thresh_cnn": 30,
"pixel_size_lateral": 108.3,
"radius": 3
}

cnn2dtest = CNNDouble_Test(config)
cnn2dtest.test()



