from plot_cloud import plot_cloud
import torch
import numpy as np
import matplotlib.pyplot as plt
checkpoint = torch.load('example_epoch_009')


xplot=range(0,64)
yplot=range(0,64)
from matplotlib.colors import Normalize
f,axs = plt.subplots(5,5)
#f2, axtest = plt.subplots(1,1)
#ax=axs[range(0,3),range(0,3)]
norm = Normalize(-35, 20)
for i in range(0,5):
    for j in range(0,5):
        string_ex = 'example' + str(i)
        rr = checkpoint[string_ex]
        rr = (rr + 1) * (55 / 2) - 35
        rr=rr.cpu()
        rr=rr.detach().numpy()
        rr=np.array(rr)
       # print(j)
       # print(rr.shape)
        '''
        test = axtest.pcolormesh(xplot, yplot, np.transpose(rr[19][0]), norm=norm)
        title_str = 'Scene' + str(i)
        axtest.set_title(title_str, fontsize=2)
        #cb = f2.colorbar(test, ax=axtest)
        '''
        rr = rr[j+15][0]

        pcm = axs[i,j].pcolormesh(xplot, yplot, np.transpose(rr), norm=norm)
        title_str = 'Scene' + str(i)
        axs[i,j].set_title(title_str, fontsize=2)
        cb = f.colorbar(pcm, ax=axs[i,j])
        cb.set_label('Reflectivities [dBZ]', fontsize=2)
        axs[i,j].tick_params(axis='both', which='major', labelsize='2')
        cb.ax.tick_params(labelsize=2)

plt.savefig('testepoch9_3')