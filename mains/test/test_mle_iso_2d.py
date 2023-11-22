import numpy as np
from tests import *
import json

config = {
"texp": 1.0,
"eta": 0.8,
"gain": "cmos/gain.npz",
"offset": "cmos/offset.npz",
"var": "cmos/var.npz",
"sigma": 0.92,
"N0": 1000,
"B0": 0,
"patchw": 3,
"lr": [1e-3,1e-3,0.0,100.0],
"max_iters": 100,
"tol": 1e-4
}

mle2dtest = MLE2DIso_Test(config)
mle2dtest.test()




