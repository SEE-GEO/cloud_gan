import datetime

import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from os import path
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# Create one file with all cloudsat_scenes
from GAN_discriminator import GAN_discriminator
from GAN_generator import GAN_generator

n_bins = 10
aspect=1/10

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device)


H_disc=[8, 256, 128, 64, 8, 9, 64, 128, 256, 256, 4096, 1]

netD = GAN_discriminator(H_disc).float().to(device)
location_of_network_parameters = './'
checkpoint = torch.load(location_of_network_parameters + 'network_parameters_CGAN_2500.pt', map_location=torch.device(device))
netD.load_state_dict(checkpoint['model_state_dict_disc'])

epoch = checkpoint['epoch']
output_folder = './early_epochs/cgan/' + str(epoch) +'/'
f1, ax1 = plt.subplots(1, 1, figsize=(10.5,15))
f2, ax2 = plt.subplots(1, 1, figsize=(10.5,15))
f3, ax3 = plt.subplots(1, 1, figsize=(10.5,15))
f4, ax4 = plt.subplots(1, 1, figsize=(10.5,15))
f5, ax5 = plt.subplots(1, 1, figsize=(10.5,15))

print('starting cs test data')
location = './modis_cloudsat_data/'
file_string = location + 'CGAN_test_data_with_temp_conc_ver2' + '.h5'
hf = h5py.File(file_string, 'r')

modis_test = torch.tensor(hf.get('modis_scenes'))
print('modis before reshape ', modis_test.shape)
modis_test = torch.cat([modis_test[:,:,:,0:3],modis_test[:,:,:,4:9]],3)
modis_test = torch.transpose(modis_test, 1, 3)
modis_test = torch.transpose(modis_test, 2, 3)
cloudsat_scenes = torch.tensor(hf.get('cloudsat_scenes'))
#cloudsat_scenes = (cloudsat_scenes + 1) * (55 / 2) - 35
#cloudsat_scenes = (cloudsat_scenes+1)*(55/2)-35
cloudsat_scenes = torch.transpose(cloudsat_scenes,2,3)
print('test set')
print('cloudsat maxvalue ',torch.max(cloudsat_scenes))
print('cloudsat shape', cloudsat_scenes.shape)
print('modis shape', modis_test.shape)


for i in range(0,10):
    index1 = (i*len(cloudsat_scenes))//10
    index2 = index1 + len(cloudsat_scenes)//10
    output_cs_temp = netD(modis_test[index1:index2].to(device),cloudsat_scenes[index1:index2].to(device)).cpu().detach().numpy()
    if i ==0:
        output_cs = output_cs_temp
    else:
        output_cs=np.concatenate((output_cs,output_cs_temp),axis=0)

print('output shape',output_cs.shape)

mean_disc = np.mean(output_cs)

ax3.text(0.5,0.91, 'Test set', transform = ax3.transAxes,fontsize = 26,color='black', horizontalalignment='center')
ax3.text(0.5,0.85,r'$\mu =$ ' +"%.4f" %(mean_disc), transform = ax3.transAxes,fontsize = 26,color='black', horizontalalignment='center')

ax3.hist(output_cs,bins = n_bins, range=(0,1), density=True)
#ax3.plot(output_cs,'.')
#ax3.set_title('test set', fontsize=32)

ax3.set_ylabel("Frequency", fontsize=28)
ax3.set_xlabel("Probability", fontsize=28)
ax3.tick_params(axis='both', which='major', labelsize=26)

ax3.set_ylim(0, 10)
ax3.set_aspect(aspect)
f3.savefig(output_folder + 'histogram_cs_test_classified_by_disc_cgan_test_' + str(epoch))
print('Finished with cs plot_test')


print('starting cs training data')

location = './modis_cloudsat_data/'
file_string = location + 'modis_cloudsat_training_data_conc_ver2' + '.h5'
hf = h5py.File(file_string, 'r')
cloudsat_scenes = torch.tensor(hf.get('cloudsat_scenes'))
cloudsat_scenes = torch.transpose(cloudsat_scenes,2,3)
#cloudsat_scenes = (cloudsat_scenes+1)*(55/2)-35
modis_test = torch.tensor(hf.get('modis_scenes'))
print('modis before reshape ', modis_test.shape)
modis_test = torch.cat([modis_test[:,:,:,0:3],modis_test[:,:,:,4:9]],3)
modis_test = torch.transpose(modis_test, 1, 3)
modis_test = torch.transpose(modis_test, 2, 3)

print('training set')
print('cloudsat maxvalue ',torch.max(cloudsat_scenes))
print('cloudsat shape', cloudsat_scenes.shape)
print('modis shape', modis_test.shape)


for i in range(0,100):
    index1 = (i*len(cloudsat_scenes))//100
    index2 = index1 + len(cloudsat_scenes)//100
    output_cs_temp = netD(modis_test[index1:index2].to(device),cloudsat_scenes[index1:index2].to(device)).cpu().detach().numpy()
    if i ==0:
        output_cs = output_cs_temp
    else:
        output_cs=np.concatenate((output_cs,output_cs_temp),axis=0)

    print('average output', np.average(output_cs))
print('output shape',output_cs.shape)

mean_disc = np.mean(output_cs)

ax4.text(0.5,0.91, 'Training set', transform = ax4.transAxes,fontsize = 26,color='black', horizontalalignment='center')
ax4.text(0.5,0.85,r'$\mu =$ ' +"%.4f" %(mean_disc), transform = ax4.transAxes,fontsize = 26,color='black', horizontalalignment='center')


ax4.hist(output_cs,bins = n_bins, range=(0,1), density=True)
#ax3.plot(output_cs,'.')
#ax4.set_title('training set', fontsize=32)
ax4.set_xlabel("Probability",fontsize = 28)
ax4.set_ylabel("Frequency", fontsize=28)
ax4.tick_params(axis='both', which='major', labelsize=26)



ax4.set_ylim(0, 10)
ax4.set_aspect(aspect)
f4.savefig(output_folder + 'histogram_cs_training_classified_by_disc_cgan_test_' + str(epoch))

print('Finished with cs plot training')



print('Starting to load generated scenes CGAN')
location = './'
file_string = location + 'CGAN_generated_scenes_for_histogram_test_temp_' +str(epoch)  + '.h5'
hf = h5py.File(file_string, 'r')

all_generated_CGAN = torch.tensor(hf.get('all_generated'))
all_generated_CGAN = all_generated_CGAN.reshape(-1,1,64,64)
all_generated_CGAN = (all_generated_CGAN +35)*2/55 - 1
#all_generated_CGAN = torch.transpose(all_generated_CGAN,2,3)
modis_generated = torch.tensor(hf.get('modis_scenes'))



print('generated set')
print('generated maxvalue ',torch.max(all_generated_CGAN))
print('generated shape', all_generated_CGAN.shape)
print('modis shape', modis_generated.shape)


for i in range(0,10):
    index1 = (i*len(all_generated_CGAN))//10
    index2 = index1 + len(all_generated_CGAN)//10

    output_cgan_temp = netD(modis_generated[index1:index2].to(device),all_generated_CGAN[index1:index2].to(device)).cpu().detach().numpy()
    if i ==0:
        output_cgan = output_cgan_temp
    else:
        output_cgan=np.concatenate((output_cgan,output_cgan_temp),axis=0)
    print('average output', np.average(output_cgan))


print('output shape',output_cgan.shape)
mean_disc = np.mean(output_cgan)

ax5.text(0.5,0.91, 'CGAN', transform = ax5.transAxes,fontsize = 26,color='black', horizontalalignment='center')
ax5.text(0.5,0.85,r'$\mu =$ ' +"%.4f" %(mean_disc), transform = ax5.transAxes,fontsize = 26,color='black', horizontalalignment='center')

ax5.hist(output_cgan,bins = n_bins, range=(0,1), density=True)
#ax3.plot(output_cs,'.')
#ax5.set_title('CGAN', fontsize=32)
ax5.tick_params(axis='both', which='major', labelsize=26)
ax5.tick_params(labelleft=True)
ax5.tick_params(labelbottom=True)

ax5.set_xlabel("Probability", fontsize=28)
ax5.set_ylabel("Frequency", fontsize=28)
ax5.set_ylim(0, 10)
ax5.set_aspect(aspect)
f5.savefig(output_folder + 'histogram_cgan_classified_by_disc_cgan_test' + str(epoch))
print('Finished with cs plot')











