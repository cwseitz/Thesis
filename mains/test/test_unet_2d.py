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
"modelname": "1204_234237",
}

unet2dtest = UNET_Test(config)
unet2dtest.test()



