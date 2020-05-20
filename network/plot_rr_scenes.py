

import numpy as np
import matplotlib.pyplot as plt

import h5py


hf = h5py.File('./rr_data//rr_data_2015_0367.h5', 'r')
rr=hf.get('rr')
rr = np.array(rr)

print(rr.shape)
rr=(rr +1)*(55/2)-35
xplot=range(0,64)
yplot=range(0,64)
from matplotlib.colors import Normalize
f,axs = plt.subplots(5,5)
#ax=axs[range(0,3),range(0,3)]
norm = Normalize(-35, 20)

for i in range(0,5):
    for j in range(0,5):
        pcm=axs[i,j].pcolormesh(xplot,yplot, np.transpose(rr[i*5 + j]), norm=norm)
        title_str = 'Scene' + str(i*5+j + 1)
        axs[i,j].set_title(title_str, fontsize=2)
        cb = f.colorbar(pcm, ax=axs[i,j])
        cb.set_label('Reflectivities [dBZ]', fontsize=2)
        axs[i,j].tick_params(axis='both', which='major', labelsize='2')
        cb.ax.tick_params(labelsize=2)
#plt.show()
