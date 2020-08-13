import numpy as np
import h5py
import geopandas
import scipy
#import libpysal
import pysal
import torch
import esda
from esda.moran import Moran
import matplotlib.pyplot as plt

hf = h5py.File('./rr_data/cloudsat_training_data_conc.h5', 'r')
hf2 = h5py.File('./IAAFT_generated_conc.h5', 'r')
cloudsat_scenes = np.array(hf.get('cloudsat_scenes'))
print('cloudsat loaded')
iaaft_scenes = np.array(hf2.get('iaaft_scenes'))
print('iaaft loaded')
print(cloudsat_scenes.shape)
print(iaaft_scenes.shape)
iaaft_scenes = np.float32(iaaft_scenes)

xplot=np.linspace(0,64*1.1,64)
yplot=np.linspace(1,16.36,64)
f, axs = plt.subplots(5,5)
f2, axs2 = plt.subplots(5,5)
print('loop started')
for i in range(0,5):
    for j in range(0,5):
        Z = cloudsat_scenes[i*5+j, 0]
        X = iaaft_scenes[i*5+j]
        w = pysal.lib.weights.lat2W(len(Z[0]), len(Z[1]), rook=True) #use rook=False to add corners to neighbours
        wx = pysal.lib.weights.lat2W(len(X[0]), len(X[1]), rook=True)
        #print(w.neighbors[12])
        mi = Moran(Z, w)
        mix = Moran(X,wx)
        #print(mi.I)
        value = "%.3f" % mi.I
        valuex ="%.3f" % mix.I
        pcm = axs[i, j].pcolormesh(xplot, yplot, np.transpose(Z))
        pcm2 = axs2[i, j].pcolormesh(xplot, yplot, np.transpose(X))
        axs[i, j].tick_params(axis='both', which='major', labelsize='7')
        axs[i, j].text(2, 8, value, fontsize=7, color='white')
        axs2[i, j].tick_params(axis='both', which='major', labelsize='7')
        axs2[i, j].text(2, 8, valuex, fontsize=7, color='white')
plt.show()

