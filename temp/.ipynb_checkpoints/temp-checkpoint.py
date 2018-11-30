import h5py
import numpy as np

f = h5py.File('vector.dat')
dset = f['Test_Vec']
x = np.ones(dset.shape)
print dset.shape
dset.read_direct(x,np.s_[:,:],np.s_[:,:])
print x