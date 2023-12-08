import numpy as np
from tests import *
import json

config = {
"modelpath": "/home/cwseitz/Desktop/Torch/Models2D/VAE1/",
"cnn_path": '/home/cwseitz/Desktop/Torch/Models2D/DeepSTORM/',
"cnn_name": '1205_040712',
"nx": 20,
"ny": 20,
"sigma": 0.92,
"N0": 1000,
"nspots" : 5,
"radius": 3,
"modelname": "1205_042709",
}

vae2dtest = VAE1_Test(config)
vae2dtest.test()



