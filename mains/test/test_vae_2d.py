import numpy as np
from tests import *
import json

config = {
"modelpath": "/home/cwseitz/Desktop/Torch/Models2D/",
"nx": 20,
"ny": 20,
"sigma": 0.92,
"N0": 1000,
"nspots" : 5,
"radius": 3,
"modelname": "1204_045829",
}

vae2dtest = VAE2D_Test(config)
vae2dtest.test()



