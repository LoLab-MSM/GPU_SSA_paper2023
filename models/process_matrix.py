import scipy.io as io
import numpy as np

d = io.loadmat('../Italy_model/k.mat')
s = io.loadmat('../Italy_model/S.mat')
print(d)
print(np.shape(d['k']))
print(d['k'][0][0])
print(s['S'])