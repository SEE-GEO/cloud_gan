from plot_cloud import plot_cloud
import torch
import numpy as np
import matplotlib.pyplot as plt
folder_path = '/cephyr/users/svcarl/Vera/cloud_gan/gan/temp_transfer/gan_training_results_ver_4/'
file_name = 'network_parameters.pt'
#folder_path = './'
#file_name = 'network_parameters_CGAN_3500.pt'
file_string = folder_path + file_name
checkpoint = torch.load(file_string, map_location=torch.device('cpu'))



num_epochs = checkpoint['epoch'] +1

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

#plt.grid(b=True)
for i in range(0,2):


    print(len(average[i]))
    if i == 0:
        f1, axs1 = plt.subplots(1, 1, figsize=(10.5, 10.5))
        axs1.plot(np.transpose(average[i]))
        title_str = string_name[i]
        axs1.set_xlabel('Epoch',fontsize=28)
        axs1.set_ylabel('Loss',fontsize=28)
        #axs1.set_title(ttle_str)

        axs1.tick_params(axis='both', which='major', labelsize=26)
        #axs1.set_aspect('equal')
        #plt.savefig('plot_loss_cgan_ver_8_gen.png')
        plt.savefig('plot_loss_gan_ver_4_gen')
        print('gen done')
    else:

        f2, axs2 = plt.subplots(1, 1, figsize=(10.5, 10.5))
        axs2.plot(np.transpose(average[i]))
        title_str = string_name[i]
        axs2.set_xlabel('Epoch',fontsize=28)
        axs2.set_ylabel('Loss',fontsize=28)
        #axs2.set_title(title_str)
        axs2.tick_params(axis='both', which='major', labelsize=26)
        #axs2.set_aspect(155 / 64)
        #plt.savefig('plot_loss_cgan_ver_8_disc.png')
        plt.savefig('plot_loss_gan_ver_4_disc')
        print('disc done')
