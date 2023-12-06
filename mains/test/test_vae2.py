import numpy as np
from tests import *
import json

config = {
"modelpath": "/home/cwseitz/Desktop/Torch/Models2D/VAE2/",
"nx": 20,
"ny": 20,
"sigma": 0.92,
"N0": 1000,
"nspots" : 5,
"radius": 3,
"modelname": "1205_040513",
}

vae2dtest = VAE2_Test(config)
vae2dtest.test()



