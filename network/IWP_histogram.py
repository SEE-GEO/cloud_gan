import h5py
from IceWaterPathMethod import IceWaterPathMethod
from GAN_generator import GAN_generator
from plot_cloud import plot_cloud
from os import path
import torch
import numpy as np
import matplotlib.pyplot as plt

# Load region below to calculate new IWP
#region
'''
location = './'
file_string = location + 'CGAN_generated_scenes_for_histogram_test_temp_.h5'
hf = h5py.File(file_string, 'r')

all_generated = torch.tensor(hf.get('all_generated'))
modis_scenes = torch.tensor(hf.get('modis_scenes'))
temperature = torch.tensor(hf.get('temperatures'))

hf2 = h5py.File('CGAN_test_data_with_temp_conc_ver2.h5','r')

cloudsat_scenes = torch.tensor(hf2.get('cloudsat_scenes'))
print('Shapes:')
print(all_generated.shape)
print(modis_scenes.shape)
print(temperature.shape)
print(cloudsat_scenes.shape)

# Take out the right amount of scenes from modis and temperatures
all_generated = all_generated.reshape(-1,1,64,64)
num_generated_scenes = int(len(all_generated))
#num_generated_scenes = 5000
#all_generated = all_generated[0:num_generated_scenes,:,:,:]

modis_scenes = modis_scenes[0:num_generated_scenes,:,:,:]
temperature = temperature[0:num_generated_scenes,:,:,:]
cloudsat_scenes = cloudsat_scenes[0:num_generated_scenes,:,:,:]
print('New Shapes:')
print(all_generated.shape)
print(modis_scenes.shape)
print(temperature.shape)
print(cloudsat_scenes.shape)

cloudsat_scenes = (cloudsat_scenes+1)*(55/2)-35

# Calculate indices for freezing level
indexes_zero = np.zeros([len(temperature),1,64,1])

for scene in range(0,len(temperature)):
    for position in range(0,64):
        for j in range(len(temperature[scene,0,position]) - 2, -1, -1):
            if (temperature[scene, 0,position,j] - 273.15) * (temperature[scene, 0, position, j + 1] - 273.15) <= 0:
                # print(i)
                # print(j)
                indexes_zero[scene,0,position] = j + 1
                break
            if j == 0:
                if torch.max(temperature[scene,0,position,:]) <= 273.15:
                    indexes_zero[scene,0,position] = 63
                else:
                    indexes_zero[scene] = 0
                    print('All above zero: ', temperature[scene,0,position])


print(indexes_zero.shape , ' shape of indices zero')
indexes_zero_for_cs = np.ones([len(indexes_zero),1,64,1])*63 - indexes_zero
print(indexes_zero_for_cs.shape , ' shape of indices zero for cs')

print(torch.max(all_generated))
print(torch.min(all_generated))

print(torch.max(cloudsat_scenes))
print(torch.min(cloudsat_scenes))


[IWP_gen_1, IWP_gen_2, IWP_gen_3] = IceWaterPathMethod(all_generated,indexes_zero_for_cs)
[IWP_cs_1, IWP_cs_2, IWP_cs_3] = IceWaterPathMethod(cloudsat_scenes,indexes_zero_for_cs)
print(IWP_gen_1.shape)
print(IWP_cs_1.shape)

IWP_gen = np.array(IWP_gen_1.reshape(-1,1))
IWP_cs = np.array(IWP_cs_1.reshape(-1,1))



hf = h5py.File('IWP_scenes_for_histogram.h5', 'w')
hf.create_dataset('IWP_gen',data=IWP_gen)
hf.create_dataset('IWP_cs', data=IWP_cs)
hf.close()
'''
#endregion

# Load calculated IWP
#region

hf = h5py.File('IWP_scenes_for_histogram.h5', 'r')
IWP_gen = np.array(hf.get('IWP_gen'))
IWP_cs = np.array(hf.get('IWP_cs'))

IWP_gen = IWP_gen*1.1*1e3
IWP_cs=IWP_cs*1.1*1e3
#endregion

textsize = 24
print(IWP_gen.shape)
print(IWP_cs.shape)

print('Max IWP generated: ', max(IWP_gen))
print('Max IWP cloudsat: ', max(IWP_cs))

standard_dev_CGAN = "%.4f" % np.std(IWP_gen)
mean_CGAN = "%.4f" % np.mean(IWP_gen)
standard_dev_cs = "%.4f" % np.std(IWP_cs)
mean_cs = "%.4f" % np.mean(IWP_cs)

print('Mean cs: ',mean_cs)
print('Mean CGAN: ',mean_CGAN)

average_cs = np.average(IWP_cs)
average_CGAN = np.average(IWP_gen)

print('Average cs: ', average_cs)
print('Average CGAN: ', average_CGAN)

# Calculate and plot histograms

n_bins = 4000
f1, axs1 = plt.subplots(1,1, figsize=(11.5,8))
f2, axs2 = plt.subplots(1,1, figsize=(11.5,8))
f3, axs3 = plt.subplots(1,1, figsize=(11.5,8))
f4, axs4 = plt.subplots(1,1, figsize=(11.5,8))

axs3.text(0.8, 0.92, 'CloudSat', transform=axs3.transAxes, fontsize=textsize, color='black')
#axs3.text( 0.8, 0.8, r'$\mu =$ '+str(mean_cs)+'\n$\sigma =$ '+str(standard_dev_cs),transform=axs3.transAxes, fontsize=22, color='grey')
axs2.text(0.82, 0.92,'CGAN', transform=axs2.transAxes, fontsize=textsize, color='black')
#axs2.text(0.8, 0.8,r'$\mu =$ '+str(mean_CGAN)+'\n$\sigma =$ '+str(standard_dev_CGAN), transform=axs2.transAxes, fontsize=22, color='grey')

axs3.text(0.77, 0.85, '--- 75 g m$^{-2}$', transform=axs3.transAxes, fontsize=textsize, color='black')
axs2.text(0.77, 0.85, '--- 75 g m$^{-2}$', transform=axs2.transAxes, fontsize=textsize, color='black')

#axs3.hist(IWP_cs, bins=n_bins,density=True, range=(0, 23), stacked = True, color='midnightblue')
axs3.hist(IWP_cs, bins=n_bins,density=True, range=(0, 26000), color='midnightblue')
axs3.minorticks_on()
axs3.tick_params(axis='both', which='major', labelsize=textsize)
axs3.tick_params(axis='both', which='minor')
axs3.set_ylabel('Frequency [m$^{2}$ g$^{-1}$]', fontsize=textsize)
axs3.set_xlabel('Ice Water Path [g m$^{-2}$]' , fontsize=textsize)
axs3.set_xlim(-50, 4500)
axs3.set_ylim(0, 0.002)
f3.tight_layout()
axs3.vlines(75,0,2,linestyles='dashed')
#axs3.vlines(average_cs,0,2,linestyles='dashed')
f3.savefig('./Results/HistogramCGAN/IWP_real_CGAN_new.png')


axs2.hist(IWP_gen, bins=n_bins, density=True, range=(0, 26000), color='midnightblue')
axs2.minorticks_on()
axs2.tick_params(axis='both', which='major', labelsize=textsize)
axs2.tick_params(axis='both', which='minor')
#axs2.set_ylabel('Frequency', fontsize=20)
axs2.set_xlabel('Ice Water Path [g m$^{-2}$]' , fontsize=textsize)
axs2.set_xlim(-50, 4500)
axs2.set_ylim(0, 0.002)
f2.tight_layout()
#axs2.vlines(average_CGAN,0,2,linestyles='dashed')
axs2.vlines(50,0,2,linestyles='dashed')
f2.savefig('./Results/HistogramCGAN/IWP_generated_CGAN_new.png')

cs_histogram, bin_edges = np.histogram(IWP_cs, bins=n_bins,range=(0, 26000),density=True)
cgan_histogram, bin_edges = np.histogram(IWP_gen, bins=n_bins, range=(0, 26000),density=True)

axs1.text(0.7, 0.92, 'CGAN-CloudSat', transform=axs1.transAxes, fontsize=textsize, color='black')

#edge_xdiff = int(np.max(bin_edges))
edge_xdiff = 26000
x_diff = np.linspace(0,edge_xdiff,n_bins)
difference = cgan_histogram - cs_histogram
axs1.plot(x_diff,difference, color = 'darkred')
axs1.minorticks_on()
axs1.tick_params(axis='both', which='major', labelsize=textsize)
axs1.tick_params(axis='both', which='minor')
axs1.set_ylabel('Frequency difference [m$^{2}$ g$^{-1}$]', fontsize=textsize)
axs1.set_xlabel('Ice Water Path [g m$^{-2}$]' , fontsize=textsize)
axs1.set_xlim(-50, 4500)
axs1.set_ylim(-0.00045,0.0001)
axs1.hlines(0,0,4500,linestyles='dashed')
f1.tight_layout()
f1.savefig('./Results/HistogramCGAN/IWP_difference_cgan_new.png')

axs4.text(0.8, 0.92, 'CloudSat', transform=axs4.transAxes, fontsize=textsize, color='black')
axs4.text( 0.8, 0.8, r'$\mu =$ '+str(mean_cs)+'\n$\sigma =$ '+str(standard_dev_cs),transform=axs4.transAxes, fontsize=textsize, color='black')
axs4.text(0.8, 0.62,'CGAN', transform=axs4.transAxes, fontsize=textsize, color='grey')
axs4.text(0.8, 0.5,r'$\mu =$ '+str(mean_CGAN)+'\n$\sigma =$ '+str(standard_dev_CGAN), transform=axs4.transAxes, fontsize=textsize, color='grey')

'''
axs4.hist(IWP_gen, bins=n_bins, range=(0, 20),density=True, color='grey')
axs4.hist(IWP_cs, bins=n_bins, range=(0, 20),density=True, color='black')
axs4.minorticks_on()
axs4.tick_params(axis='both', which='major', labelsize='22')
axs4.tick_params(axis='both', which='minor')
axs4.set_ylabel('Frequency', fontsize=22)
axs4.set_xlabel('Ice Water Path [g m$^{-2}$]' , fontsize=22)
axs4.set_xlim(-0.2, 6)
axs4.set_ylim(0, 2.1)
f4.savefig('./Results/HistogramCGAN/IWP_real_and_gen_CGAN.png')
'''