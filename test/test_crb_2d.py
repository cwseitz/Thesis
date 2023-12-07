from .tests import CRB2D_Test1
import matplotlib.pyplot as plt
import json

config = 
{
"texp": 1.0,
"nx": 50,
"ny": 50,
"eta": 0.8,
"gain": "cmos/gain.npz",
"offset": "cmos/offset.npz",
"var": "cmos/var.npz",
"N0": 10000,
"B0": 0,
"pixel_size_lateral": 108.3,
"x0": 25,
"y0": 25,
"z0": 0.0,
"lr": [1e-5,1e-5,1e-5,100.0]
}

with open('iso2d.json', 'r') as f:
    setup_params = json.load(f)
crb2dtest = CRB2D_Test1(setup_params)
fig,ax = plt.subplots()
crb2dtest.plot(ax)
plt.show()
