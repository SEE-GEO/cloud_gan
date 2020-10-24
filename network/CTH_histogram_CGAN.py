import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from os import path
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# Create one file with all cloudsat_scenes
import matplotlib.ticker as ticker

from GAN_generator import GAN_generator

n_bins = 64

'''
location = './'
file_string = location + 'CGAN_generated_scenes_for_histogram_test_temp_.h5'
hf = h5py.File(file_string, 'r')
all_generated = torch.tensor(hf.get('all_generated'))

hf2 = h5py.File('CGAN_test_data_with_temp_conc_ver2.h5','r')
cloudsat_scenes = torch.tensor(hf2.get('cloudsat_scenes'))

print('Shapes:')
print(all_generated.shape)
print(cloudsat_scenes.shape)
cloudsat_scenes = (cloudsat_scenes + 1) * (55 / 2) - 35
cloudsat_scenes = cloudsat_scenes.view(-1, 64)
print('Shape of cloudsat_scenes', cloudsat_scenes.shape)
'''
#Load region below to calculate new altitudes
#region
'''
altitudes = np.zeros(len(cloudsat_scenes))
#altitudes = np.zeros(1000)
altitudes_gen = np.zeros(len(all_generated))
for scene in range(0,len(cloudsat_scenes)):
#for scene in range(0,1000):
    first_cloud_location = 0
    for j in range(63, -1, -1): #Loop backwards, starting from top of scene
        if cloudsat_scenes[scene,j] >= -20 and first_cloud_location == 0: #Set dBZ limit for cloud top detection
            altitudes[scene] = j #save the index of the altitude where the cloud top is for each position
            first_cloud_location =+1

for scene in range(0, len(all_generated)):
    # for scene in range(0,1000):
    first_cloud_location_gen = 0
    for j in range(63, -1, -1):  # Loop backwards, starting from top of scene
        if all_generated[scene,j] >=-20 and first_cloud_location_gen ==0:
            altitudes_gen[scene] = j
            first_cloud_location_gen =+1

altitudes = (altitudes*0.24)+1 #Altitude of cloud top over sea level, [km]
altitudes_gen = (altitudes_gen*0.24)+1
print('Shape of altitudes arrays:', altitudes.shape, altitudes_gen)

hf = h5py.File('CGAN_altitudes_for_histogram.h5', 'w')
hf.create_dataset('altitudes', data=altitudes)
hf.create_dataset('altitudes_gen', data=altitudes_gen)
hf.close()
'''
#endregion

hf = h5py.File('CGAN_altitudes_for_histogram.h5', 'r')
altitudes = np.array(hf.get('altitudes'))
altitudes_gen = np.array(hf.get('altitudes_gen'))

print(altitudes_gen.shape, ' shape altitudes gen')
print(altitudes.shape,' shape altitudes cs')

standard_dev_CGAN = "%.4f" % np.std(altitudes_gen)
mean_CGAN = "%.4f" % np.mean(altitudes_gen)
standard_dev_cs = "%.4f" % np.std(altitudes)
mean_cs = "%.4f" % np.mean(altitudes)

print(mean_cs)
print(mean_CGAN)

average_cgan = np.average(altitudes_gen)
average_cs =  np.average(altitudes)
print('AVERAGES')
print(average_cs)
print(average_cgan)



f1, axs1 = plt.subplots(1,1, figsize=(11.5,8))
f2, axs2 = plt.subplots(1,1, figsize=(11.5,8))
f3, axs3 = plt.subplots(1,1, figsize=(11.5,8))

axs3.text(0.02, 0.92, 'CloudSat', transform=axs3.transAxes, fontsize=22, color='black')
axs3.text( 0.02, 0.8, r'$\mu =$ '+str(mean_cs)+',\n$\sigma =$ '+str(standard_dev_cs),transform=axs3.transAxes, fontsize=22, color='grey')
axs2.text(0.02, 0.92,'CGAN', transform=axs2.transAxes, fontsize=22, color='black')
axs2.text(0.02, 0.8,r'$\mu =$ '+str(mean_CGAN)+',\n$\sigma =$ '+str(standard_dev_CGAN), transform=axs2.transAxes, fontsize=22, color='grey')


axs3.hist(altitudes, bins=n_bins, range=(1, 16.36), density=True, color='midnightblue')
axs3.minorticks_on()
axs3.tick_params(axis='both', which='major', labelsize='22')
axs3.tick_params(axis='both', which='minor')
axs3.set_ylabel('Frequency [km$^{-1}$]', fontsize=22)
axs3.set_ylim(0, 0.165)
#axs3.set_xlim(0,16.36)
axs3.vlines(average_cs,0,0.165,linestyles='dashed')
#axs3.vlines(8.55,0,0.27,linestyles='dashed')
axs3.set_xlabel('Cloud-top height [km]' , fontsize=22)
f3.savefig('./Results/HistogramCGAN/CTH_real_CGAN.png')


axs2.hist(altitudes_gen, bins=n_bins, range=(1, 16.36), density=True, color='midnightblue')
axs2.minorticks_on()
axs2.tick_params(axis='both', which='major', labelsize='22')
axs2.tick_params(axis='both', which='minor')
axs2.set_ylabel('Frequency [km$^{-1}$]', fontsize=22) #For presentation
axs2.set_ylim(0, 0.165)
#axs2.set_xlim(0,16.36)
axs2.vlines(average_cgan,0,0.165,linestyles='dashed')
#axs2.vlines(14.7,0,0.27,linestyles='dashed')
#axs2.vlines(16.2,0,0.27,linestyles='dashed')

#axs2.set_ylabel('Frequency', fontsize=20)
axs2.set_xlabel('Cloud-top height [km]' , fontsize=22)
f2.savefig('./Results/Presentation/CTH_generated_CGAN.png')

cs_histogram, bin_edges = np.histogram(altitudes, bins=n_bins, range=(1, 16.36), density=True)
cgan_histogram, bin_edges = np.histogram(altitudes_gen, bins=n_bins, range=(1, 16.36), density=True)

axs1.text(0.7005, 0.92, 'CGAN-CloudSat', transform=axs1.transAxes, fontsize=22, color='black')

x_diff = np.linspace(1,16.36,n_bins)
difference =  cgan_histogram - cs_histogram
axs1.plot(x_diff,difference, color = 'darkred')
axs1.minorticks_on()
axs1.tick_params(axis='both', which='major', labelsize='22')
axs1.tick_params(axis='both', which='minor')
axs1.set_ylabel('Frequency difference [km$^{-1}$]', fontsize=20)
axs1.set_xlabel('Cloud-top height [km]' , fontsize=22)
axs1.set_ylim(-0.05,0.05)
axs1.hlines(0,1,16.36,linestyles='dashed')
f1.tight_layout()
f1.savefig('./Results/Presentation/CTH_difference_cgan.png')
