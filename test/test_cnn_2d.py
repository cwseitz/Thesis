import numpy as np
from tests import *
import json

config = {
"modelpath": "/home/cwseitz/Desktop/Torch/Models2D/DeepSTORM/",
"modelname": "1206_150420",
"nx": 20,
"ny": 20,
"disc_radius": 5.0,
"nspots": 5,
"kwargs":{
"texp": 1.0,
"eta": 0.5,
"sigma": 0.92,
"N0": 100.0,
"B0": 1.0,
"gain": 1.0,
"offset": 0.0,
"var": 0.0
},
"thresh_cnn":30,
"radius":3,
"pixel_size_lateral": 108.3,
}

cnn2dtest = CNNDouble_Test(config)
cnn2dtest.test()



