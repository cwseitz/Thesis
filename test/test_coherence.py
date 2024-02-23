import numpy as np
from tests import *
import json

config = {
"modelpath": "/home/cwseitz/Desktop/Torch/Models/DeepSTORM/",
"nx": 20,
"ny": 20,
"kwargs":{
"texp": 1.0,
"eta": 0.5,
"sigma": 0.55,
"N00": 0,
"N01": 100,
"B0": 10,
"gain": 1.0,
"offset": 0.0,
"var": 0.0,
"nframes":100,
"sigma_ring": 0.39,
},
"ring_radius": 0.8,
"nspots" : 3,
"modelname": "0124_070823",
"thresh_cnn": 30,
"pixel_size_lateral": 108.3,
"radius": 3
}

test = CNNCoherence_Test(config)
test.test()



