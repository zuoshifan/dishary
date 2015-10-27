import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


with h5py.File('Cas_gain_full.hdf5', 'r') as f:
    LM = f['/xx/LM'][...]

L, M = LM[:, 0], LM[:, 1]
fig = plt.figure()
fig.add_subplot(111,aspect='equal')
plt.plot(L, M)
plt.ylim(-0.15, 0.15)
plt.axhline(0, color='k', linestyle='--')
plt.axvline(0, color='k', linestyle='--')
thetas = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] # degree
for theta in thetas:
    theta = np.radians(theta)
    l = np.linspace(-np.sqrt(1.0 - np.cos(theta)**2), np.sqrt(1.0 - np.cos(theta)**2), 100)
    mplus = np.sqrt(1.0 - l**2 - np.cos(theta)**2)
    mminus = -np.sqrt(1.0 - l**2 - np.cos(theta)**2)
    plt.plot(l, mplus, '--k')
    plt.plot(l, mminus, '--k')
plt.xlabel('$l$')
plt.ylabel('$m$')
plt.savefig('LM.png')