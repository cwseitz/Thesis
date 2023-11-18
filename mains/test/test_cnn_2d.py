import numpy as np
from tests import *
import json

config = {
"modelpath": "/home/cwseitz/Desktop/Torch/Models2D/",
"modelname": "0731_015007",
"pixel_size_lateral": 108.3,
"thresh_cnn": 30,
"radius": 4
}

cnn2dtest = CNN2D_Test(config)
cnn2dtest.test()



