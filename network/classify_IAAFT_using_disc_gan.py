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

n_bins = 100
n_removed = int((30 / 165) * n_bins)

location = './'
file_string = location + 'IAAFT_generated_scenes_GAN_testset' + '.h5'
hf = h5py.File(file_string, 'r')
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device)
iaaft_scenes = torch.tensor(hf.get('iaaft_scenes'))
iaaft_scenes_temp = torch.ones([len(iaaft_scenes),1,64,64])
print(iaaft_scenes_temp.shape)
iaaft_scenes_temp[:,0,:,:] = iaaft_scenes[:,:,:]
iaaft_scenes=iaaft_scenes_temp
iaaft_scenes = torch.transpose(iaaft_scenes,2,3)
print(iaaft_scenes.shape)
print(iaaft_scenes[0:10])

H_disc=[5, 256, 128, 128, 5, 1, 64, 128, 256, 256, 4096, 1]
H_gen=[384,16384, 256, 128, 64, 1]
netG = GAN_generator(H_gen).float().to(device)
netD = GAN_discriminator(H_disc).float().to(device)
location_of_network_parameters = './gan_training_results_ver_4/'
checkpoint = torch.load(location_of_network_parameters + 'network_parameters.pt', map_location=torch.device('cpu'))
netD.load_state_dict(checkpoint['model_state_dict_disc'])
netG.load_state_dict(checkpoint['model_state_dict_gen'])
b_size = len(iaaft_scenes)//10
D_in_gen = [b_size, 64, 6]

noise = (torch.randn(D_in_gen)).to(device)
generated = netG(noise, None)
output_generated = netD(None,generated)
output_iaaft = netD(None,iaaft_scenes)
output_generated = output_generated.detach().numpy()
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.hist(output_generated,bins = n_bins,density=True)
plt.savefig('histogram_generated_gan_classified_by_disc_gan_test')

print(output_iaaft.shape)
print(torch.mean(output_iaaft))
output_iaaft = output_iaaft.detach().numpy()
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
ax2.hist(output_iaaft,bins = n_bins,density=True)
plt.savefig('histogram_iaaft_classified_by_disc_gan_test')