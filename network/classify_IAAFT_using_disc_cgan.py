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
checkpoint = torch.load(location_of_network_parameters + 'network_parameters_CGAN_3500.pt', map_location=torch.device(device))
netD.load_state_dict(checkpoint['model_state_dict_disc'])


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
modis_test = torch.cat([modis_test[:,:,:,0:3],modis_test[:,:,:,4:9]],3)
modis_test = torch.transpose(modis_test, 1, 3)
modis_test = torch.transpose(modis_test, 2, 3)
cloudsat_scenes = torch.tensor(hf.get('cloudsat_scenes'))
#cloudsat_scenes = (cloudsat_scenes + 1) * (55 / 2) - 35
#cloudsat_scenes = (cloudsat_scenes+1)*(55/2)-35

for i in range(0,10):
    index1 = (i*len(cloudsat_scenes))//10
    index2 = index1 + len(cloudsat_scenes)//10
    print(cloudsat_scenes[index1:index2].shape)
    print(modis_test[index1:index2].shape)
    output_cs_temp = netD(modis_test[index1:index2].to(device),cloudsat_scenes[index1:index2].to(device)).cpu().detach().numpy()
    if i ==0:
        output_cs = output_cs_temp
    else:
        output_cs=np.concatenate((output_cs,output_cs_temp),axis=0)

    print(np.average(output_cs))
    print(torch.max(cloudsat_scenes))
    print(torch.min(cloudsat_scenes))



ax3.hist(output_cs,bins = n_bins, range=(0,1), density=True)
#ax3.plot(output_cs,'.')
ax3.set_ylabel("Frequency", fontsize=28)
ax3.set_xlabel("Probability", fontsize=28)
ax3.tick_params(axis='both', which='major', labelsize=26)

ax3.set_ylim(0, 10)
ax3.set_aspect(aspect)
f3.savefig('histogram_cs_test_classified_by_disc_cgan_test')
print('Finished with cs plot_training')


print('starting cs training data')

location = './modis_cloudsat_data/'
file_string = location + 'modis_cloudsat_training_data_conc_ver2' + '.h5'
hf = h5py.File(file_string, 'r')
cloudsat_scenes = torch.tensor(hf.get('cloudsat_scenes'))
#cloudsat_scenes = (cloudsat_scenes+1)*(55/2)-35
modis_test = torch.tensor(hf.get('modis_scenes'))
modis_test = torch.cat([modis_test[:,:,:,0:3],modis_test[:,:,:,4:9]],3)
modis_test = torch.transpose(modis_test, 1, 3)
modis_test = torch.transpose(modis_test, 2, 3)


for i in range(0,100):
    index1 = (i*len(cloudsat_scenes))//100
    index2 = index1 + len(cloudsat_scenes)//100
    print(cloudsat_scenes[index1:index2].shape)
    print(modis_test[index1:index2].shape)
    output_cs_temp = netD(modis_test[index1:index2].to(device),cloudsat_scenes[index1:index2].to(device)).cpu().detach().numpy()
    if i ==0:
        output_cs = output_cs_temp
    else:
        output_cs=np.concatenate((output_cs,output_cs_temp),axis=0)

    print(np.average(output_cs))
    print(torch.max(cloudsat_scenes))
    print(torch.min(cloudsat_scenes))



print(np.average(output_cs))
print(torch.max(cloudsat_scenes))
print(torch.min(cloudsat_scenes))


ax4.hist(output_cs,bins = n_bins, range=(0,1), density=True)
#ax3.plot(output_cs,'.')
ax4.set_xlabel("Probability",fontsize = 28)
ax4.set_ylabel("Frequency", fontsize=28)
ax4.tick_params(axis='both', which='major', labelsize=26)



ax4.set_ylim(0, 10)
ax4.set_aspect(aspect)
f4.savefig('histogram_cs_training_classified_by_disc_cgan_test')

print('Finished with cs plot training')



print('Starting to load generated scenes CGAN')
location = './'
file_string = location + 'CGAN_generated_scenes_for_histogram_test_temp_' + '.h5'
hf = h5py.File(file_string, 'r')

all_generated_CGAN = torch.tensor(hf.get('all_generated'))
all_generated_CGAN = all_generated_CGAN.reshape(-1,1,64,64)
all_generated_CGAN = (all_generated_CGAN +35)*2/55 - 1

modis_generated = torch.tensor(hf.get('modis_scenes'))




for i in range(0,10):
    index1 = (i*len(all_generated_CGAN))//10
    index2 = index1 + len(all_generated_CGAN)//10

    output_cgan_temp = netD(modis_test[index1:index2].to(device),all_generated_CGAN[index1:index2].to(device)).cpu().detach().numpy()
    if i ==0:
        output_cgan = output_cgan_temp
    else:
        output_cgan=np.concatenate((output_cgan,output_cgan_temp),axis=0)


ax5.hist(output_cgan,bins = n_bins, range=(0,1), density=True)
#ax3.plot(output_cs,'.')
ax5.set_title('CGAN', fontsize=32)
ax5.tick_params(axis='both', which='major', labelsize=26)
ax5.tick_params(labelleft=True)
ax5.tick_params(labelbottom=True)

ax5.set_xlabel("Probability", fontsize=28)
ax5.set_ylim(0, 10)
ax5.set_aspect(aspect)
f5.savefig('histogram_cgan_classified_by_disc_cgan_test')
print('Finished with cs plot')











