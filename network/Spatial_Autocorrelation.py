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

num_bins = 1000
# Load data to calculate Morans I
#region


hf = h5py.File('./rr_data/cloudsat_test_data_conc.h5', 'r')
#hf2 = h5py.File('./IAAFT_generated_scenes_GAN_testset.h5', 'r')
#hfGAN_old = h5py.File('./Results/HistogramGAN/GAN_generated_scenes_for_histogram.h5','r') #OLD!
hfGAN = h5py.File('./GAN_generated_scenes_for_histogram_400.h5','r')
GAN_scenes = np.array(hfGAN.get('all_generated'))
GAN_scenes = GAN_scenes.reshape(-1,1,64,64)
print('GAN loaded')
cloudsat_scenes = np.array(hf.get('cloudsat_scenes'))
print('cloudsat loaded')
#iaaft_scenes = np.array(hf2.get('iaaft_scenes'))
#print('iaaft loaded')
print(GAN_scenes.shape)
print(cloudsat_scenes.shape)
#print(iaaft_scenes.shape)
#iaaft_scenes = np.float32(iaaft_scenes)

#iaaft_scenes = (iaaft_scenes+1)*(55/2)-35
cloudsat_scenes = (cloudsat_scenes+1)*(55/2)-35
GAN_scenes = (GAN_scenes+1)*(55/2)-35

xplot=np.linspace(0,64*1.1,64)
yplot=np.linspace(1,16.36,64)
f, axs = plt.subplots(5,5)
f2, axs2 = plt.subplots(5,5)
f3, axs3 = plt.subplots(2,1, figsize=(6.5,7.5))
f4, axs4 = plt.subplots(1,1, figsize=(6.5,4))


total_mi_template = np.zeros([len(cloudsat_scenes),1])
#total_mi_iaaft = np.zeros([len(iaaft_scenes),1])
total_mi_GAN = np.zeros([len(GAN_scenes),1])

w = pysal.lib.weights.lat2W(64,64, rook=True) #Use rook = False to include neighbors across diagonals


#endregion

# Calculate Moran's I for CloudSat template and IAAFT below
#region
'''
for i in range(0,len(cloudsat_scenes)):
    #print('i value ', i, 'cs scene ',cloudsat_scenes[i,0])
    if np.all(cloudsat_scenes[i,0] == cloudsat_scenes[i,0,0,0]): #Check if all values in the array are equal:
        total_mi_template[i,0] = 4444
    else:
        mi = Moran(cloudsat_scenes[i,0],w)
        total_mi_template[i,0] = mi.I
    if i % 500==0:
        print(i)
    #print(mi.I)
for j in range(0,len(iaaft_scenes)):
    #print('j value ', j, 'iaaft scene ',iaaft_scenes[j])
    if np.all(iaaft_scenes[j] == iaaft_scenes[j, 0, 0]):
        total_mi_iaaft[j,0] = 4444
    else:
        mi = Moran(iaaft_scenes[j],w)
        total_mi_iaaft[j,0] = mi.I
    if j%500==0:
        print(j)
    #print(mi.I)


total_mi_template = total_mi_template[total_mi_template!= 4444]
total_mi_iaaft = total_mi_iaaft[total_mi_iaaft!=4444]

hf = h5py.File('Morans_I_for_histogram_IAAFT.h5', 'w')
hf.create_dataset('total_mi_iaaft',data=total_mi_iaaft)
hf.create_dataset('total_mi_template', data=total_mi_template)
hf.close()
'''
#endregion
# Load calculated Moran's I for IAAFT
#region
'''
hf = h5py.File('Morans_I_for_histogram_IAAFT.h5', 'r')
total_mi_iaaft = np.array(hf.get('total_mi_iaaft'))
total_mi_template = np.array(hf.get('total_mi_template'))
'''
#endregion

# Plot histograms IAAFT
#region
'''
f3, axs3 = plt.subplots(2,1, figsize=(6.5,7.5))
f4, axs4 = plt.subplots(1,1, figsize=(6.5,4))

standard_dev_template = "%.4f" % np.std(total_mi_template)
standard_dev_iaaft = "%.4f" % np.std(total_mi_iaaft)
mean_template = "%.4f" % np.mean(total_mi_template)
mean_iaaft = "%.4f" % np.mean(total_mi_iaaft)

max_iaaft = "%.4f" % np.max(total_mi_iaaft)
min_iaaft = "%.4f" % np.min(total_mi_iaaft)
max_cs = "%.4f" % np.max(total_mi_template)
min_cs = "%.4f" % np.min(total_mi_template)

average_cs = np.average(total_mi_template)
average_iaaft= np.average(total_mi_iaaft)

axs3[0].text(0.02, 0.9, 'CloudSat template', transform=axs3[0].transAxes, fontsize=12, color='black')
axs3[0].text( 0.02, 0.63, r'$\mu =$ '+str(mean_template)+'\n$\sigma =$ '+str(standard_dev_template)+ '\n$I_{max} =$ '+str(max_cs)+ '\n$I_{min} =$ '+str(min_cs),transform=axs3[0].transAxes, fontsize=12, color='grey')
axs3[1].text(0.02, 0.9,'IAAFT', transform=axs3[1].transAxes, fontsize=12, color='black')
axs3[1].text(0.02, 0.63,r'$\mu =$ '+str(mean_iaaft)+'\n$\sigma =$ '+str(standard_dev_iaaft)+ '\n$I_{max} =$ '+str(max_iaaft)+ '\n$I_{min} =$ '+str(min_iaaft), transform=axs3[1].transAxes, fontsize=12, color='grey')

axs3[0].hist(total_mi_template, bins=num_bins, range=(-1, 1), density=True, color='midnightblue')
axs3[0].minorticks_on()
axs3[0].tick_params(axis='both', which='major', labelsize='12')
axs3[0].tick_params(axis='both', which='minor')

axs3[1].hist(total_mi_iaaft, bins=num_bins, range=(-1, 1), density=True, color='midnightblue')
axs3[1].minorticks_on()
axs3[1].tick_params(axis='both', which='major', labelsize='12')
axs3[1].tick_params(axis='both', which='minor')

axs3[1].set_ylabel('\nFrequency ($I^{-1}$)', fontsize=12)
axs3[0].set_ylabel('\nFrequency ($I^{-1}$)', fontsize=12)
axs3[1].set_xlabel('Moran\'s $I$' , fontsize=12)
#axs3[0].set_xlabel('Moran\'s I (CloudSat template)', fontsize=12)
axs3[0].set_xlim(0.6,1)
axs3[1].set_xlim(0.6,1)
axs3[0].set_ylim(0,19)
axs3[1].set_ylim(0,19)
axs3[1].vlines(average_iaaft,0,19,linestyles='dashed')
axs3[0].vlines(average_cs,0,19,linestyles='dashed')
f3.tight_layout()
f3.savefig('./Results/IAAFT/spatial_hist_iaaft.png')

cs_histogram, bin_edges = np.histogram(total_mi_template, bins=num_bins, range=(-1, 1), density=True)
iaaft_histogram, bin_edges = np.histogram(total_mi_iaaft, bins=num_bins, range=(-1, 1), density=True)
axs4.text(0.02, 0.9, 'IAAFT-CloudSat', transform=axs4.transAxes, fontsize=12, color='black')

x_diff = np.linspace(-1,1,num_bins)
difference =  iaaft_histogram - cs_histogram
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
f4.savefig('./Results/IAAFT/spatial_hist_difference_iaaft.png')
'''
#endregion

# Calculate Morans I for GAN
#region

for i in range(0,len(GAN_scenes)):
    if np.all(GAN_scenes[i,0] == GAN_scenes[i,0,0,0]): #Check if all values in the array are equal:
        total_mi_GAN[i,0] = 4444
    else:
        mi = Moran(GAN_scenes[i,0],w)
        total_mi_GAN[i,0] = mi.I
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
total_mi_GAN= total_mi_GAN[total_mi_GAN!= 4444]
total_mi_template= total_mi_template[total_mi_template!= 4444]

hf = h5py.File('Morans_I_for_histogram_GAN_400.h5', 'w')
hf.create_dataset('total_mi_GAN',data=total_mi_GAN)
hf.create_dataset('total_mi_template', data=total_mi_template)
hf.close()

#endregion

# Load calculated Moran's I for GAN
#region
'''
hf = h5py.File('Morans_I_for_histogram_GAN_400.h5', 'r')
total_mi_GAN = np.array(hf.get('total_mi_GAN'))
total_mi_template = np.array(hf.get('total_mi_template'))
'''
#endregion

# Plot histograms for GAN
#region

standard_dev_GAN = "%.4f" % np.std(total_mi_GAN)
mean_GAN = "%.4f" % np.mean(total_mi_GAN)
standard_dev_cs = "%.4f" % np.std(total_mi_template)
mean_cs = "%.4f" % np.mean(total_mi_template)

max_GAN = "%.4f" % np.max(total_mi_GAN)
min_GAN = "%.4f" % np.min(total_mi_GAN)
max_cs = "%.4f" % np.max(total_mi_template)
min_cs = "%.4f" % np.min(total_mi_template)

average_cs = np.average(total_mi_template)
average_GAN = np.average(total_mi_GAN)

f3, axs3 = plt.subplots(2,1, figsize=(7,7.5))
f4, axs4 = plt.subplots(1,1, figsize=(6.5,4))

axs3[1].text(0.02, 0.9, 'GAN', transform=axs3[1].transAxes, fontsize=12, color='black')
axs3[1].text( 0.02, 0.63, r'$\mu =$ '+str(mean_GAN)+'\n$\sigma =$ '+str(standard_dev_GAN)+ '\n$I_{max} =$ '+str(max_GAN)+ '\n$I_{min} =$ '+str(min_GAN),transform=axs3[1].transAxes, fontsize=12, color='grey')
axs3[0].text(0.02, 0.9,'CloudSat', transform=axs3[0].transAxes, fontsize=12, color='black')
axs3[0].text(0.02, 0.63,r'$\mu =$ '+str(mean_cs)+'\n$\sigma =$ '+str(standard_dev_cs)+ '\n$I_{max} =$ '+str(max_cs)+ '\n$I_{min} =$ '+str(min_cs), transform=axs3[0].transAxes, fontsize=12, color='grey')

axs3[1].hist(total_mi_GAN, bins=num_bins, range=(-1, 1), density=True, color='midnightblue')
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
axs3[1].vlines(average_GAN,0,19,linestyles='dashed')
axs3[0].vlines(average_cs,0,19,linestyles='dashed')
f3.tight_layout()
f3.savefig('./Results/HistogramGAN/spatial_hist.png')

cs_histogram, bin_edges = np.histogram(total_mi_template, bins=num_bins, range=(-1, 1), density=True)
gan_histogram, bin_edges = np.histogram(total_mi_GAN, bins=num_bins, range=(-1, 1), density=True)
axs4.text(0.02, 0.9, 'GAN-CloudSat', transform=axs4.transAxes, fontsize=12, color='black')

x_diff = np.linspace(-1,1,num_bins)
difference =  gan_histogram - cs_histogram
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
f4.savefig('./Results/HistogramGAN/spatial_hist_difference_gan.png')

#endregion


'''
print('loop started')
for i in range(0,5):
    for j in range(0,5):
        #Z = cloudsat_scenes[i*5+j, 0]
        #X = iaaft_scenes[i*5+j,0]
        Y = GAN_scenes[i*5+j,0]
        #print(w.neighbors[12])
        #mi = Moran(Z, w)
        #mix = Moran(X,w)
        miy = Moran(Y,w)
        #print(mi.I)
        #value = "%.3f" % mi.I
        #valuex ="%.3f" % mix.I
        valuey = "%.3f" % miy.I
        pcm = axs[i, j].pcolormesh(xplot, yplot, np.transpose(Y))
        #pcm2 = axs2[i, j].pcolormesh(xplot, yplot, np.transpose(X))
        axs[i, j].tick_params(axis='both', which='major', labelsize='7')
        axs[i, j].text(2, 15, valuey, fontsize=7, color='white')
        #axs2[i, j].tick_params(axis='both', which='major', labelsize='7')
        #axs2[i, j].text(2, 15, valuex, fontsize=7, color='white')
plt.show()

'''