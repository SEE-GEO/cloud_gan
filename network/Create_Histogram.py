
import h5py
import torch
import  numpy as np
import matplotlib.pyplot as plt
from os import path
#from mpl_toolkits.axes_grid1 import make_axes_locatable

#Create one file with all cloudsat_scenes
import matplotlib.ticker as ticker

from GAN_generator import GAN_generator

def fmt(x,pos):
    a,b= '{:.2e}'.format(x).split('e')
    b=int(b)
    #return r'${} \times 10^{{{}}}$'.format(a,b)
    return r'${} $'.format(a,b)


n_bins = 55*4
n_removed = int((30/165)*n_bins)

location = './rr_data/'
file_string = location + 'cloudsat_test_data_conc' + '.h5'
hf = h5py.File(file_string, 'r')

cloudsat_scenes = torch.tensor(hf.get('cloudsat_scenes'))
cloudsat_scenes = (cloudsat_scenes+1)*(55/2)-35
cloudsat_scenes = cloudsat_scenes.view(-1,64)
print('Shape of cloudsat_scenes', cloudsat_scenes.shape)
dbz_per_bin = 55/n_bins
total_histogram = np.ones([64,n_bins])
for i in range(0,64):
    scenes_histogram, bin_edges = np.histogram(cloudsat_scenes[:,i], bins=n_bins, range=(-35,20), density=False)
    total_histogram[i] = scenes_histogram
    total_sum_real = np.sum(total_histogram[i])

    total_histogram[i] = total_histogram[i] / (dbz_per_bin * total_sum_real)
print(total_histogram.shape)
#print(bin_edges)
#total_sum_real = np.sum(total_histogram)

#total_histogram = total_histogram/(dbz_per_bin*total_sum_real)
new_total = total_histogram[:,n_removed:n_bins]
print(new_total.shape)


#D_gen = [50, 64, 6]

#Load this (below) if you want to create new generated scenes
'''
D_gen = [len(cloudsat_scenes)//(64*100), 64, 6]
H_gen=[384,16384, 256, 128, 64, 1]
netG = GAN_generator(H_gen)
folder = './gan_training_results_ver_4/'
if path.exists(folder + 'network_parameters.pt'):
    
    with torch.no_grad():

        checkpoint = torch.load(folder + 'network_parameters.pt', map_location=torch.device('cpu'))
        netG.load_state_dict(checkpoint['model_state_dict_gen'])
        netG.zero_grad()
        print('dadada')
        all_generated = []
        for i in range(0,100):
            generated_scenes=netG(torch.randn(D_gen), None)
            print('size of generated scenes: ',generated_scenes.shape)
            generated_scenes = generated_scenes.view(-1, 64)
            generated_scenes = generated_scenes.detach().numpy()
            generated_scenes = np.array(generated_scenes)
            print(i)
            if i == 0 :
                all_generated = generated_scenes
            else:
                all_generated = np.append(all_generated,generated_scenes,axis=0)

        all_generated = (all_generated + 1) * (55 / 2) - 35

    name_string = 'generated_scenes_for_histogram'
    filename = 'GAN_' + name_string + '.h5'
    hf = h5py.File(filename, 'w')
    hf.create_dataset('all_generated', data=all_generated)
    hf.close()
'''

print('Starting to load generated scenes')
location = './'
file_string = location + 'GAN_generated_scenes_for_histogram' + '.h5'
hf = h5py.File(file_string, 'r')

all_generated = torch.tensor(hf.get('all_generated'))

print('Generated scenes loaded')

total_histogram_generated = np.ones([64,n_bins])

for i in range(0, 64):
    scenes_histogram, bin_edges = np.histogram(all_generated[:, i], bins=n_bins, range=(-35, 20), density=False)
    total_histogram_generated[i] = scenes_histogram
    total_sum_real = np.sum(total_histogram_generated[i])

    total_histogram_generated[i] = total_histogram_generated[i] / (dbz_per_bin * total_sum_real)

#total_sum_generated = np.sum(total_histogram_generated)
#total_histogram_generated = total_histogram_generated /( dbz_per_bin * total_sum_generated)
new_total_generated = total_histogram_generated[:, n_removed:n_bins]

occurence_diff = new_total_generated - new_total
percentage_diff = np.zeros([64,len(new_total[0])])

for i in range(0,64):
    for j in range(0,len(new_total[0])):
        if new_total[i,j] == 0:
            percentage_diff[i,j]=np.nan
        else:
            percentage_diff[i,j] = occurence_diff[i,j]/new_total[i,j]

x_ocean=np.linspace(bin_edges[n_removed],20,n_bins-n_removed)
y_ocean=np.linspace(1,16.36,64)
#y=np.linspace(1,13,50) #240*50+1000
f1,ax1 = plt.subplots(1, 1, figsize=(10.5,15))
pcm1 = ax1.pcolormesh(x_ocean, y_ocean, new_total, vmin=0, vmax=0.035)
ax1.set_ylabel("Altitude [km]", fontsize=28)
ax1.set_xlabel("Reflectivity [dBZ]", fontsize=28)
#ax.set_aspect((num_bins-num_removed_bins)/64)
ax1.set_aspect(155/64)
#ax1.set_aspect('auto')
#ax1.set_title('Ocean', fontsize=14)
cb1=f1.colorbar(pcm1,ax=ax1, orientation = 'horizontal', fraction =0.049, pad=0.15)
cb1.set_label('Norm. occurrence (Real) [dBZ$^{-1}$]', fontsize=28)
ax1.tick_params(axis='both', which='major', labelsize=26)
#cb1.set_ticks(np.arange(np.min(new_total), np.max(new_total), step=0.01))
cb1.ax.tick_params(labelsize=26, rotation=45)
#cb1.ax.tick_params(labelsize=12)
#f1.tight_layout()
ax1.set_title('GAN', fontsize = 32)
plt.savefig('Histogram_GAN_real')
ax1.set_title('IAAFT',fontsize = 32)
plt.savefig('Histogram_IAAFT_real')

x_land=np.linspace(bin_edges[n_removed],20,n_bins-n_removed)
y_land=np.linspace(1,16.36,64)
#y=np.linspace(1,13,50) #240*50+1000
f2,ax2 = plt.subplots(1, 1, figsize=(10.5,15))
pcm2 = ax2.pcolormesh(x_land, y_land, new_total_generated, vmin=0, vmax=0.035)
#ax2.set_ylabel("Altitude", fontsize=14)
ax2.set_ylabel("Altitude [km]", fontsize=28)
ax2.set_xlabel("Reflectivity [dBZ]", fontsize=28)
#ax.set_aspect((num_bins-num_removed_bins)/64)
ax2.set_aspect(155/64)
#ax2.set_aspect('auto')
#ax2.set_title('Land', fontsize=14)
cb2=f2.colorbar(pcm2,ax=ax2, orientation = 'horizontal', fraction =0.049, pad=0.15)
cb2.set_label('Norm. occurrence (GAN) [dBZ$^{-1}$]', fontsize=28)
ax2.tick_params(axis='both', which='major', labelsize=26)
#cb2.set_ticks(np.arange(np.min(new_total_generated), np.max(new_total_generated), step=0.01))
cb2.ax.tick_params(labelsize=26, rotation=45)
#f2.tight_layout()
plt.savefig('Histogram_GAN_generated')
#plt.show()

#Occurence difference histogram below
#occurence_diff = new_total - new_total_generated
x_diff=np.linspace(bin_edges[n_removed],20,n_bins-n_removed)
y_diff=np.linspace(1,16.36,64)
#y=np.linspace(1,13,50) #240*50+1000
f3,ax3 = plt.subplots(1, 1, figsize=(10.5,15))
pcm3 = ax3.pcolormesh(x_diff, y_diff, occurence_diff,vmin = -0.017, vmax = 0.017,cmap= 'seismic')
#ax3.set_ylabel("Altitude", fontsize=14)
ax3.set_ylabel("Altitude [km]", fontsize=28)
ax3.set_xlabel("Reflectivity [dBZ]", fontsize=28)
#ax.set_aspect((num_bins-num_removed_bins)/64)
ax3.set_aspect(155/64)
#ax3.set_aspect('auto')
#ax3.set_title('Occurence difference')
cb3=f3.colorbar(pcm3,ax=ax3, orientation = 'horizontal', fraction =0.049, pad=0.15)
cb3.set_label('Occurrence bias (GAN - Real) [dBZ$^{-1}$]', fontsize=28)
ax3.tick_params(axis='both', which='major', labelsize=26)
#cb3.set_ticks(np.arange(-0.0075, 0.01, step=0.0025))
cb3.ax.tick_params(labelsize=26, rotation = 45)
#f3.tight_layout()
plt.savefig('Difference_histogram_GAN_real_and_generated')

f4,ax4 = plt.subplots(1, 1, figsize=(10.5,15))
pcm4 = ax4.pcolormesh(x_diff, y_diff, percentage_diff, vmin =-2, vmax = 2,cmap= 'seismic')
#ax3.set_ylabel("Altitude", fontsize=14)
ax4.set_xlabel("Reflectivity [dBZ]", fontsize=28)
#ax.set_aspect((num_bins-num_removed_bins)/64)
#ax4.set_aspect(135/64)
ax4.set_aspect('auto')
#ax3.set_title('Occurence difference')
cb4=f4.colorbar(pcm4,ax=ax4, orientation = 'horizontal', fraction =0.049, pad=0.15)
cb4.set_label('Occurrence bias (GAN - Real)/Real', fontsize=28)
ax4.tick_params(axis='both', which='major', labelsize=26)
#cb3.set_ticks(np.arange(-0.0075, 0.01, step=0.0025))
cb4.ax.tick_params(labelsize=26, rotation = 45)
#f3.tight_layout()
plt.savefig('Percentage_histogram_GAN_real_and_generated')

#plt.show