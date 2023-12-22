import numpy as np

nx,ny = 20,20
gain = 1.0*np.ones((nx,ny))
offset = 100*np.ones((nx,ny))
var = 5*np.ones((nx,ny))

np.savez('gain.npz',gain)
np.savez('offset.npz',offset)
np.savez('var.npz',var)
