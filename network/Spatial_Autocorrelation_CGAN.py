import numpy as np
import h5py
import geopandas
import scipy
#import libpysal
import pysal
import torch
import esda
from esda.moran import Moran
import matplotlib.pyplot as plt


# Load data to calculate Morans I
#region
'''
location = './'
file_string = location + 'CGAN_generated_scenes_for_histogram_test_temp_.h5'
hf = h5py.File(file_string, 'r')
CGAN_scenes = np.array(hf.get('all_generated'))

hf2 = h5py.File('CGAN_test_data_with_temp_conc_ver2.h5','r')
cloudsat_scenes = np.array(hf2.get('cloudsat_scenes'))
CGAN_scenes = CGAN_scenes.reshape(-1,1,64,64)

print('Shapes:')
print(CGAN_scenes.shape)
print(cloudsat_scenes.shape)
cloudsat_scenes = (cloudsat_scenes + 1) * (55 / 2) - 35

print('Maximums:')
print(np.max(cloudsat_scenes))
print(np.max(CGAN_scenes))

total_mi_template = np.zeros([len(cloudsat_scenes),1])
total_mi_CGAN = np.zeros([len(CGAN_scenes),1])

w = pysal.lib.weights.lat2W(64,64, rook=True) #Use rook = False to include neighbors across diagonals

#endregion

# Calculate Morans I for CGAN
#region

for i in range(0,len(CGAN_scenes)):
    if np.all(CGAN_scenes[i,0] == CGAN_scenes[i,0,0,0]): #Check if all values in the array are equal:
        total_mi_CGAN[i,0] = 4444
    else:
        mi = Moran(CGAN_scenes[i,0],w)
        total_mi_CGAN[i,0] = mi.I
    if i % 500==0:
        print(i)
for i in range(0,len(cloudsat_scenes)):
    if np.all(cloudsat_scenes[i,0] == cloudsat_scenes[i,0,0,0]): #Check if all values in the array are equal:
        total_mi_template[i,0] = 4444
    else:
        mi = Moran(cloudsat_scenes[i,0],w)
        total_mi_template[i,0] = mi.I
    if i % 500==0:
        print(i)
total_mi_GAN= total_mi_CGAN[total_mi_CGAN!= 4444]
total_mi_template= total_mi_template[total_mi_template!= 4444]

hf = h5py.File('Morans_I_for_histogram_CGAN.h5', 'w')
hf.create_dataset('total_mi_CGAN',data=total_mi_CGAN)
hf.create_dataset('total_mi_template', data=total_mi_template)
hf.close()
'''
#endregion

# Load calculated Moran's I for GAN
#region

hf = h5py.File('Morans_I_for_histogram_CGAN.h5', 'r')
total_mi_CGAN = np.array(hf.get('total_mi_CGAN'))
total_mi_template = np.array(hf.get('total_mi_template'))

#endregion

# Plot histograms for GAN
#region

num_bins = 1000

xplot=np.linspace(0,64*1.1,64)
yplot=np.linspace(1,16.36,64)
f, axs = plt.subplots(5,5)
f2, axs2 = plt.subplots(5,5)
f3, axs3 = plt.subplots(2,1, figsize=(7,7.5))
f4, axs4 = plt.subplots(1,1, figsize=(6.5,4))

standard_dev_CGAN = "%.4f" % np.std(total_mi_CGAN)
mean_CGAN = "%.4f" % np.mean(total_mi_CGAN)
standard_dev_cs = "%.4f" % np.std(total_mi_template)
mean_cs = "%.4f" % np.mean(total_mi_template)

max_CGAN = "%.4f" % np.max(total_mi_CGAN)
min_CGAN = "%.4f" % np.min(total_mi_CGAN)
max_cs = "%.4f" % np.max(total_mi_template)
min_cs = "%.4f" % np.min(total_mi_template)

average_cs = np.average(total_mi_template)
average_CGAN = np.average(total_mi_CGAN)

axs3[1].text(0.02, 0.9, 'CGAN', transform=axs3[1].transAxes, fontsize=12, color='black')
axs3[1].text( 0.02, 0.63, r'$\mu =$ '+str(mean_CGAN)+'\n$\sigma =$ '+str(standard_dev_CGAN)+ '\n$I_{max} =$ '+str(max_CGAN)+ '\n$I_{min} =$ '+str(min_CGAN),transform=axs3[1].transAxes, fontsize=12, color='grey')
axs3[0].text(0.02, 0.9,'CloudSat', transform=axs3[0].transAxes, fontsize=12, color='black')
axs3[0].text(0.02, 0.63,r'$\mu =$ '+str(mean_cs)+'\n$\sigma =$ '+str(standard_dev_cs)+ '\n$I_{max} =$ '+str(max_cs)+ '\n$I_{min} =$ '+str(min_cs), transform=axs3[0].transAxes, fontsize=12, color='grey')

axs3[1].hist(total_mi_CGAN, bins=num_bins, range=(-1, 1), density=True, color='midnightblue')
axs3[1].minorticks_on()
axs3[1].tick_params(axis='both', which='major', labelsize='12')
axs3[1].tick_params(axis='both', which='minor')

axs3[0].hist(total_mi_template, bins=num_bins, range=(-1, 1), density=True, color='midnightblue')
axs3[0].minorticks_on()
axs3[0].tick_params(axis='both', which='major', labelsize='12')
axs3[0].tick_params(axis='both', which='minor')

axs3[1].set_ylabel('\nFrequency ($I^{-1}$)', fontsize=12)
axs3[0].set_ylabel('\nFrequency ($I^{-1}$)', fontsize=12)
axs3[1].set_xlabel('Moran\'s $I$' , fontsize=12)
#axs3[0].set_xlabel('Moran\'s I (CloudSat template)', fontsize=12)
axs3[0].set_xlim(0.6,1)
axs3[1].set_xlim(0.6,1)
axs3[0].set_ylim(0,19)
axs3[1].set_ylim(0,19)
axs3[1].vlines(average_CGAN,0,19,linestyles='dashed')
axs3[0].vlines(average_cs,0,19,linestyles='dashed')
f3.tight_layout()
f3.savefig('./Results/HistogramCGAN/spatial_hist_cgan.png')

cs_histogram, bin_edges = np.histogram(total_mi_template, bins=num_bins, range=(-1, 1), density=True)
cgan_histogram, bin_edges = np.histogram(total_mi_CGAN, bins=num_bins, range=(-1, 1), density=True)
axs4.text(0.02, 0.9, 'CGAN-CloudSat', transform=axs4.transAxes, fontsize=12, color='black')

x_diff = np.linspace(-1,1,num_bins)
difference =  cgan_histogram - cs_histogram
axs4.plot(x_diff,difference, color = 'darkred')
axs4.minorticks_on()
axs4.tick_params(axis='both', which='major', labelsize='12')
axs4.tick_params(axis='both', which='minor')
axs4.set_ylabel('Frequency difference ($I^{-1}$)', fontsize=12)
axs4.set_xlabel('Moran\'s $I$' , fontsize=12)
axs4.hlines(0,0.6,1,linestyles='dashed')
axs4.set_ylim(-6.5,6.5)
axs4.set_xlim(0.6,1)
f4.tight_layout()
f4.savefig('./Results/HistogramCGAN/spatial_hist_difference_cgan.png')

#endregion
