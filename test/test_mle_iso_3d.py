import numpy as np
import json
from .tests import MLE3D_Test

config = 
{
"texp": 1.0,
"nx": 50,
"ny": 50,
"eta": 0.8,
"gain": "cmos/gain.npz",
"offset": "cmos/offset.npz",
"var": "cmos/var.npz",
"N0": 5000,
"B0": 0,
"pixel_size_lateral": 108.3,
"x0": 25,
"y0": 25,
"z0": 1.0
}
    
mle3dtest = MLE3D_Test(setup_params)
mle3dtest.test()

