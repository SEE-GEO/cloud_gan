from plot_cloud import plot_cloud
import torch
import numpy as np
import matplotlib.pyplot as plt
#folder_path = '/cephyr/users/svcarl/Vera/cloud_gan/gan/temp_transfer/training_results_old/'
folder_path = './'
file_name = 'network_parameters_CGAN.pt'
file_string = folder_path + file_name
checkpoint = torch.load(file_string, map_location=torch.device('cpu'))


from matplotlib.colors import Normalize
f,axs = plt.subplots(1,2)
#f2, axtest = plt.subplots(1,1)
#ax=axs[range(0,3),range(0,3)]
num_epochs = 55

string_name = ['loss_gen','loss_disc']
average = np.zeros((2,num_epochs))
for i in range(0, 2):
    loss = checkpoint[string_name[i]]

    loss = np.array(loss)
    num_losses = len(loss)
    for j in range(0,num_epochs):
        average[i,j]=0
        for k in range (0,int(num_losses/num_epochs)):
            average[i,j] = loss[j*int(num_losses/num_epochs) + k]/int(num_losses/num_epochs) + average[i,j]

plt.grid(b=True)
for i in range(0,2):


    print(len(average[i]))

    pcm = axs[i].plot(np.transpose(average[i]))
    title_str = string_name[i]
    axs[i].set_xlabel('epoch')
    axs[i].set_ylabel('loss')
    axs[i].set_title(title_str)
    axs[i].tick_params(axis='both', which='major')

plt.savefig('plot_loss_cgan_ver_1_training2.png')