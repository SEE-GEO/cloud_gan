import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from os import path
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# Create one file with all cloudsat_scenes
import matplotlib.ticker as ticker

from GAN_generator import GAN_generator


def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a,b)
    #return r'${} $'.format(a, b)


n_bins = 64

location = './rr_data/'
file_string = location + 'cloudsat_test_data_conc' + '.h5'
hf = h5py.File(file_string, 'r')

cloudsat_scenes = torch.tensor(hf.get('cloudsat_scenes'))
cloudsat_scenes = (cloudsat_scenes + 1) * (55 / 2) - 35
cloudsat_scenes = cloudsat_scenes.view(-1, 64)
print('Shape of cloudsat_scenes', cloudsat_scenes.shape)

hf2 = h5py.File('./IAAFT_generated_scenes_GAN_testset.h5', 'r')
iaaft_scenes = np.array(hf2.get('iaaft_scenes'))
print('IAAFT scenes ',iaaft_scenes.shape)
iaaft_scenes = iaaft_scenes.reshape(-1,64)
iaaft_scenes = (iaaft_scenes+1)*(55/2)-35


#Load region below to calculate new altitudes
#region
'''
altitudes = np.zeros(len(cloudsat_scenes))
#altitudes = np.zeros(1000)
altitudes_gen = np.zeros(len(iaaft_scenes))
for scene in range(0,len(cloudsat_scenes)):
#for scene in range(0,1000):
    first_cloud_location = 0
    for j in range(63, -1, -1): #Loop backwards, starting from top of scene
        if cloudsat_scenes[scene,j] >= -20 and first_cloud_location == 0: #Set dBZ limit for cloud top detection
            altitudes[scene] = j #save the index of the altitude where the cloud top is for each position
            first_cloud_location =+1

for scene in range(0, len(iaaft_scenes)):
    # for scene in range(0,1000):
    first_cloud_location_gen = 0
    for j in range(63, -1, -1):  # Loop backwards, starting from top of scene
        if iaaft_scenes[scene,j] >=-20 and first_cloud_location_gen ==0:
            altitudes_gen[scene] = j
            first_cloud_location_gen =+1

altitudes = (altitudes*0.24)+1 #Altitude of cloud top over sea level, [km]
altitudes_gen = (altitudes_gen*0.24)+1
print('Shape of altitudes arrays:', altitudes.shape, altitudes_gen)

hf = h5py.File('IAAFT_altitudes_for_histogram.h5', 'w')
hf.create_dataset('altitudes', data=altitudes)
hf.create_dataset('altitudes_gen', data=altitudes_gen)
hf.close()
'''
#endregion

hf = h5py.File('IAAFT_altitudes_for_histogram.h5', 'r')
altitudes = np.array(hf.get('altitudes'))
altitudes_gen = np.array(hf.get('altitudes_gen'))


standard_dev_IAAFT = "%.4f" % np.std(altitudes_gen)
mean_IAAFT = "%.4f" % np.mean(altitudes_gen)
standard_dev_cs = "%.4f" % np.std(altitudes)
mean_cs = "%.4f" % np.mean(altitudes)

print(mean_cs)
print(mean_IAAFT)

average_gen = np.average(altitudes_gen)
average_cs =  np.average(altitudes)

f1, axs1 = plt.subplots(1,1, figsize=(11.5,8))
f2, axs2 = plt.subplots(1,1, figsize=(11.5,8))
f3, axs3 = plt.subplots(1,1, figsize=(11.5,8))
f4, axs4 = plt.subplots(1,1, figsize=(11.5,8))

axs3.text(0.02, 0.92, 'CloudSat', transform=axs3.transAxes, fontsize=22, color='black')
axs3.text( 0.02, 0.8, r'$\mu =$ '+str(mean_cs)+',\n$\sigma =$ '+str(standard_dev_cs),transform=axs3.transAxes, fontsize=22, color='grey')
axs2.text(0.02, 0.92,'IAAFT', transform=axs2.transAxes, fontsize=22, color='black')
axs2.text(0.02, 0.8,r'$\mu =$ '+str(mean_IAAFT)+',\n$\sigma =$ '+str(standard_dev_IAAFT), transform=axs2.transAxes, fontsize=22, color='grey')


axs3.hist(altitudes, bins=n_bins, range=(1, 16.36), density=True, stacked =True,color='midnightblue')
axs3.minorticks_on()
axs3.tick_params(axis='both', which='major', labelsize='22')
axs3.tick_params(axis='both', which='minor')
axs3.set_ylabel('Frequency [km$^{-1}$]', fontsize=22)
axs3.set_ylim(0, 0.165)
axs3.vlines(average_cs,0,0.165,linestyles='dashed')
axs3.set_xlabel('Cloud-top height [km]' , fontsize=22)
f3.savefig('./Results/IAAFT/CTH_real_iaaft.png')


axs2.hist(altitudes_gen, bins=n_bins, range=(1, 16.36), density=True, stacked =True,color='midnightblue')
axs2.minorticks_on()
axs2.tick_params(axis='both', which='major', labelsize='22')
axs2.tick_params(axis='both', which='minor')
axs2.set_ylabel('Frequency [km$^{-1}$]', fontsize=22) #For presentation
axs2.set_ylim(0, 0.165)
axs2.vlines(average_gen,0,0.165,linestyles='dashed')
axs2.set_xlabel('Cloud-top height [km]' , fontsize=22)
f2.savefig('./Results/Presentation/CTH_generated_iaaft.png')

cs_histogram, bin_edges = np.histogram(altitudes, bins=n_bins, range=(1, 16.36), density=True)
iaaft_histogram, bin_edges = np.histogram(altitudes_gen, bins=n_bins, range=(1, 16.36), density=True)

axs1.text(0.73, 0.92, 'IAAFT-CloudSat', transform=axs1.transAxes, fontsize=22, color='black')

x_diff = np.linspace(1,16.36,n_bins)
difference = iaaft_histogram - cs_histogram
axs1.plot(x_diff,difference, color = 'darkred')
axs1.minorticks_on()
axs1.tick_params(axis='both', which='major', labelsize='22')
axs1.tick_params(axis='both', which='minor')
axs1.set_ylabel('Frequency difference [km$^{-1}$]', fontsize=20)
axs1.set_xlabel('Cloud-top height [km]' , fontsize=22)
axs1.hlines(0,1,16.36,linestyles='dashed')
axs1.set_ylim(-0.05,0.05)
f1.tight_layout()
f1.savefig('./Results/Presentation/CTH_difference_iaaft.png')

axs4.text(0.8, 0.92, 'CloudSat', transform=axs4.transAxes, fontsize=22, color='black')
axs4.text( 0.8, 0.8, r'$\mu =$ '+str(mean_cs)+'\n$\sigma =$ '+str(standard_dev_cs),transform=axs4.transAxes, fontsize=22, color='black')
axs4.text(0.8, 0.62,'IAAFT', transform=axs4.transAxes, fontsize=22, color='grey')
axs4.text(0.8, 0.5,r'$\mu =$ '+str(mean_IAAFT)+'\n$\sigma =$ '+str(standard_dev_IAAFT), transform=axs4.transAxes, fontsize=22, color='grey')

axs4.hist(altitudes, bins=n_bins, range=(1, 16.36),density=True, color='black')
axs4.hist(altitudes_gen, bins=n_bins, range=(1, 16.36),density=True, color='grey')
axs4.minorticks_on()
axs4.tick_params(axis='both', which='major', labelsize='22')
axs4.tick_params(axis='both', which='minor')
axs4.set_ylabel('Frequency [km$^{-1}$]', fontsize=22)
axs4.set_xlabel('Ice Water Path [g m$^{-2}$]' , fontsize=22)
axs4.set_ylim(0, 0.165)
f4.savefig('./Results/IAAFT/CTH_real_and_iaaft.png')
