import numpy as np
import numpy.ma as ma
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import constants
import h5py
import torch

hf = h5py.File('./rr_data/cloudsat_test_data_conc.h5', 'r')

'''
iaaft_scenes = []
counter = 0
for i in range(1,9):
    hf2 = h5py.File('./IAAFT_data/IAAFT_generated_scenes_GAN_'+str(i)+'.h5', 'r')
    iaaft_scenes_temp = np.array(hf2.get('iaaft_scenes'))
    if counter ==0:
        iaaft_scenes = iaaft_scenes_temp
        counter =+1
    else:
        iaaft_scenes = np.concatenate((iaaft_scenes,iaaft_scenes_temp),axis=0)

filename = 'IAAFT_generated_conc.h5'
hf_new = h5py.File(filename, 'w')
hf_new.create_dataset('iaaft_scenes', data=iaaft_scenes)
'''

def fmt(x,pos):
    a,b='{:.2e}'.format(x).split('e')
    b=int(b)
    #return r'${} \times 10^{{{}}}$'.format(a,b)
    return r'${} $'.format(a, b)

hf2 = h5py.File('./IAAFT_generated_conc.h5', 'r')
iaaft_scenes = np.array(hf2.get('iaaft_scenes'))
print('IAAFT scenes ',iaaft_scenes.shape)

cloudsat = hf.get('cloudsat_scenes')
cloudsat = np.array(cloudsat)
print('CloudSat shape ', cloudsat.shape)

iaaft_scenes = np.float32(iaaft_scenes)
print('Data type IAAFT ', iaaft_scenes.dtype)
print('Data type CloudSat ', cloudsat.dtype)

iaaft_scenes = (iaaft_scenes+1)*(55/2)-35
cloudsat = (cloudsat+1)*(55/2)-35

#Below are flattening operations in order to create the histograms
iaaft_scenes = iaaft_scenes.reshape(-1,64)
cloudsat = cloudsat.reshape(-1,64)

num_bins = 55*4
num_removed = int((30/165)*num_bins)
total_iaaft_histogram = np.ones([64,num_bins])
total_cloudsat_histogram = np.ones([64,num_bins])

dBz_per_bin = 55/num_bins

for i in range(0,64):
    iaaft_histogram, bin_edges_iaaft = np.histogram(iaaft_scenes[:, i], bins=num_bins, range=(-35, 20), density=False)
    cloudsat_histogram, bin_edges_cloudsat = np.histogram(cloudsat[:, i], bins=num_bins, range=(-35, 20), density=False)
    sum_cs = np.sum(cloudsat_histogram)
    sum_iaaft = np.sum(iaaft_histogram)
    total_iaaft_histogram[i] = iaaft_histogram/(dBz_per_bin*sum_iaaft)
    total_cloudsat_histogram[i] = cloudsat_histogram/(dBz_per_bin*sum_cs)
print('IAAFT histogram shape ',total_iaaft_histogram.shape)
print('CloudSat histogram shape ',total_cloudsat_histogram.shape)

new_total_iaaft = total_iaaft_histogram[:,num_removed:num_bins]
print(new_total_iaaft.shape)
new_total_cloudsat = total_cloudsat_histogram[:,num_removed:num_bins]


occurence_diff = new_total_iaaft - new_total_cloudsat

print('cloudsat hist shape ',new_total_cloudsat.shape)
print('occ diff shape ',occurence_diff.shape)
percentage_diff =np.zeros([64,len(new_total_cloudsat[0])])


zero_counter = 0
for i in range(0,64):
    for j in range(0,len(new_total_cloudsat[0])):
        if new_total_cloudsat[i,j] == 0:
            #print('i and j: ', i, j, 'Occ diff: ',occurence_diff[i,j])
            #zero_counter=zero_counter+1
            #percentage_diff[i,j] = 1
        #elif new_total_cloudsat[i,j] == 0 and occurence_diff[i,j] >= 1e-5:
            zero_counter=zero_counter+1
            percentage_diff[i,j] = np.nan
            #print('NaN percentage: ',percentage_diff[i])
        else:
            percentage_diff[i,j] = occurence_diff[i,j]/new_total_cloudsat[i,j]
large_counter = 0
for i in range(0,64):
    for j in range(0,len(new_total_cloudsat[0])):
        if abs(percentage_diff[i,j]) > 2:
            #print('Large percentage diff at i:', i, ' j: ',j, ' Value: ', percentage_diff[i,j])
            large_counter = large_counter +1

print('Number of zeros in cloudsat: ',zero_counter)
print('Number of large percentages: ',large_counter)
print('percentage hist shape ',percentage_diff.shape)
#print(percentage_diff[0])



#Plot histograms in section below
#region

x_hist=np.linspace(bin_edges_iaaft[num_removed],20,num_bins-num_removed)
y_hist=np.linspace(1,16.36,64)

f1,ax1 = plt.subplots(1, 1, figsize=(10.5,15))
pcm1 = ax1.pcolormesh(x_hist, y_hist, new_total_iaaft, vmax = 0.035)
#ax1.set_ylabel("Altitude [km]", fontsize=28)
ax1.set_xlabel("Reflectivity [dBZ]", fontsize=28)
#ax.set_aspect((num_bins-num_removed_bins)/64)
ax1.set_aspect(155/64)
#ax1.set_title('IAAFT', fontsize=28)
cb1=f1.colorbar(pcm1,ax=ax1, orientation = 'horizontal', fraction =0.049, pad=0.15)
cb1.set_label('Norm. occurrence (IAAFT) [dBZ$^{-1}$]', fontsize=28)
ax1.tick_params(axis='both', which='major', labelsize=26)
#cb1.set_ticks(np.arange(np.min(new_total_iaaft), np.max(new_total_iaaft), step=0.01))
cb1.ax.tick_params(labelsize=26, rotation=45)
#cb1.ax.tick_params(labelsize=12)
#f1.tight_layout()
plt.savefig('./Results/IAAFT/IAAFT_histogram_GAN.png')


f2,ax2 = plt.subplots(1, 1, figsize=(10.5,15))
pcm2 = ax2.pcolormesh(x_hist, y_hist, new_total_cloudsat, vmax = 0.035)
ax2.set_ylabel("Altitude [km]", fontsize=28)
ax2.set_xlabel("Reflectivity [dBZ]", fontsize=28)
#ax.set_aspect((num_bins-num_removed_bins)/64)
ax2.set_aspect(155/64)
#ax2.set_title('Real', fontsize=28)
cb2=f2.colorbar(pcm2,ax=ax2, orientation = 'horizontal', fraction =0.049, pad=0.15)
cb2.set_label('Norm. occurrence (Real) [dBZ$^{-1}$]', fontsize=28)
ax2.tick_params(axis='both', which='major', labelsize=26)
#cb2.set_ticks(np.arange(np.min(new_total_cloudsat), np.max(new_total_cloudsat), step=0.01))
cb2.ax.tick_params(labelsize=26, rotation=45)
#f2.tight_layout()
plt.savefig('./Results/IAAFT/Cloudsat_histogram_GAN_for_IAAFT.png')


#Occurrence difference histogram below
#occurence_diff = new_total_cloudsat - new_total_iaaft
#percentage_diff = np.true_divide(occurence_diff,new_total_cloudsat)

x_diff=np.linspace(bin_edges_cloudsat[num_removed],20,num_bins-num_removed)
y_diff=np.linspace(1,16.36,64)
f3,ax3 = plt.subplots(1, 1, figsize=(10.5,15))
pcm3 = ax3.pcolormesh(x_diff, y_diff, occurence_diff,vmin = -0.017, vmax = 0.017, cmap= 'seismic')
#ax3.set_ylabel("Altitude", fontsize=28)
ax3.set_xlabel("Reflectivity [dBZ]", fontsize=28)
#ax.set_aspect((num_bins-num_removed_bins)/64)
ax3.set_aspect(155/64)
#ax3.set_title('Occurence difference', fontsize=28)
cb3=f3.colorbar(pcm3,ax=ax3, orientation = 'horizontal', fraction =0.049, pad=0.15)
cb3.set_label('Occurrence bias (IAAFT - Real) [dBZ$^{-1}$]', fontsize=28)
ax3.tick_params(axis='both', which='major', labelsize=26)
#cb3.set_ticks(np.arange(-0.0075, 0.01, step=0.0025))
cb3.ax.tick_params(labelsize=26, rotation = 45)
#f3.tight_layout()

plt.savefig('./Results/IAAFT/Difference_histogram_IAAFTGAN.png')
#endregion

'''
f4,ax4 = plt.subplots(1, 1, figsize=(10.5,15))
pcm4 = ax4.pcolormesh(x_diff, y_diff, percentage_diff, cmap= 'tab20c')
#ax3.set_ylabel("Altitude", fontsize=28)
ax4.set_xlabel("Reflectivity [dBZ]", fontsize=28)
#ax.set_aspect((num_bins-num_removed_bins)/64)
ax4.set_aspect(135/64)
#ax3.set_title('Occurence difference', fontsize=28)
cb4=f4.colorbar(pcm4,ax=ax4, orientation = 'horizontal', fraction =0.049, pad=0.15, format=ticker.FuncFormatter(fmt))
cb4.set_label('Relative occurence bias (Real - IAAFT)/Real', fontsize=28)
ax4.tick_params(axis='both', which='major', labelsize=26)
cb4.set_ticks(np.arange(-10, 1, step=1))
cb4.ax.tick_params(labelsize=26, rotation = 45)
#f3.tight_layout()
plt.savefig('./Results/IAAFT/Percentage_histogram_IAAFTGAN.png')

f5,ax5 = plt.subplots(1, 1, figsize=(10.5,15))
pcm5 = ax5.pcolormesh(x_diff, y_diff, percentage_diff, vmin = -2, vmax = 2, cmap= 'seismic')
#ax3.set_ylabel("Altitude", fontsize=28)
ax5.set_xlabel("Reflectivity [dBZ]", fontsize=28)
#ax.set_aspect((num_bins-num_removed_bins)/64)
ax5.set_aspect(135/64)
#ax3.set_title('Occurence difference', fontsize=28)
cb5=f5.colorbar(pcm5,ax=ax5, orientation = 'horizontal', fraction =0.049, pad=0.15, format=ticker.FuncFormatter(fmt))
cb5.set_label('Relative occurence bias (Real - IAAFT)/Real', fontsize=28)
ax5.tick_params(axis='both', which='major', labelsize=26)
#cb5.set_ticks(np.arange(-10, 1, step=1))
cb5.ax.tick_params(labelsize=26, rotation = 45)
#f3.tight_layout()
plt.savefig('./Results/IAAFT/Percentage_histogram_IAAFTGAN_seismic.png')
'''
#plt.show()

