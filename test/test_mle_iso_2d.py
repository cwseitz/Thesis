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
"max_iters": 100,
}

mle2dtest = MLE2D_BFGS_Test(config)
mle2dtest.test(niters=100)




