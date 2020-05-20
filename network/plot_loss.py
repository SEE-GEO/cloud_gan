from plot_cloud import plot_cloud
import torch
import numpy as np
import matplotlib.pyplot as plt
checkpoint = torch.load('network_parameters.pt')


from matplotlib.colors import Normalize
f,axs = plt.subplots(1,2)
#f2, axtest = plt.subplots(1,1)
#ax=axs[range(0,3),range(0,3)]
string_name = ['loss_gen','loss_disc']
for i in range(0,2):

    loss = checkpoint[string_name[i]]

    loss=np.array(loss)


    pcm = axs[i].plot(np.transpose(loss))
    title_str = string_name[i]
    axs[i].set_xlabel('batch_number')
    axs[i].set_ylabel('loss')
    axs[i].set_title(title_str)
    axs[i].tick_params(axis='both', which='major')

plt.show()