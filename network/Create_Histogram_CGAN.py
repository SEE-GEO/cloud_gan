import datetime

import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from os import path
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# Create one file with all cloudsat_scenes
from GAN_generator import GAN_generator
import matplotlib.ticker as ticker

def fmt(x,pos):
    a,b= '{:.2e}'.format(x).split('e')
    b=int(b)
    #return r'${} \times 10^{{{}}}$'.format(a,b)
    return r'${} $'.format(a,b)


n_bins = 4*55
n_removed = int((30 / 165) * n_bins)

location = './modis_cloudsat_data/'
file_string = location + 'modis_cloudsat_test_data_conc_ver2' + '.h5'
hf = h5py.File(file_string, 'r')

cloudsat_scenes = torch.tensor(hf.get('cloudsat_scenes'))
cloudsat_scenes = (cloudsat_scenes + 1) * (55 / 2) - 35
cloudsat_scenes = cloudsat_scenes.view(-1, 64)
modis_scenes = torch.tensor(hf.get('modis_scenes'))
print('shape of mdis scenes: ',modis_scenes.shape)

print(len(cloudsat_scenes)," files loaded: ")
temp_modis_scenes = torch.cat([modis_scenes[:,:,:,0:3],modis_scenes[:,:,:,4:9]],3)
modis_scenes=temp_modis_scenes

print('Shape of cloudsat_scenes', cloudsat_scenes.shape)

total_histogram = np.ones([64, n_bins])

for i in range(0, 64):
    scenes_histogram, bin_edges = np.histogram(cloudsat_scenes[:, i], bins=n_bins, range=(-35, 20), density=False)
    total_histogram[i] = scenes_histogram
print(total_histogram.shape)

total_sum_real = np.sum(total_histogram)
dbz_per_bin = 55/n_bins
total_histogram = total_histogram/(dbz_per_bin*total_sum_real)

# print(bin_edges)
new_total = total_histogram[:, n_removed:n_bins]
print(new_total.shape)

x = np.linspace(bin_edges[n_removed], 20, n_bins - n_removed)
y = np.linspace(1, 16.36, 64)
f, ax = plt.subplots(1, 1, figsize=(10.5,15))

pcm = ax.pcolormesh(x, y, new_total)
ax.set_ylabel("Altitude [km]", fontsize=28)
ax.set_xlabel("Reflectivity [dBZ]", fontsize=28)
#ax.set_aspect((n_bins - n_removed) / 64)
ax.set_aspect(135/64)
ax.tick_params(axis='both', which='major', labelsize=26)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes('bottom', size='100%', pad=0.05)
# cax = f.add_axes([0.09, 0.06, 0.84, 0.02])
cb = f.colorbar(pcm, ax=ax, orientation='horizontal', fraction=0.049, pad=0.15, format=ticker.FuncFormatter(fmt))
cb.set_label('Norm. occurrence (Real) $10^{-4}$ [dBZ$^{-1}$]', fontsize=28)
cb.ax.tick_params(labelsize=26, rotation=45)
plt.savefig('histogram_real_CGAN')

# D_gen = [50, 64, 6]

# Load this (below) if you want to create new generated scenes

D_gen = [len(cloudsat_scenes)//(64*100),1, 64, 1]
H_gen = [576, 16384, 256, 128, 64, 1]
netG = GAN_generator(H_gen)
folder = './'

modis_scenes = torch.transpose(modis_scenes, 1, 3)
modis_scenes = torch.transpose(modis_scenes, 2, 3)
print('så här ser modis ut: ',modis_scenes.shape)
if path.exists(folder + 'network_parameters_CGAN_2650.pt'):

    with torch.no_grad():

        checkpoint = torch.load(folder + 'network_parameters_CGAN_2650.pt', map_location=torch.device('cpu'))
        netG.load_state_dict(checkpoint['model_state_dict_gen'])
        netG.zero_grad()
        print('dadada')
        all_generated = []
        for i in range(0,100):
            print('Test shape ',(modis_scenes[i*len(cloudsat_scenes)//(64*100):(i+1)*len(cloudsat_scenes)//(64*100), :,:, :]).shape)
            print((torch.randn(D_gen)).shape)

            generated_scenes = netG(torch.randn(D_gen), modis_scenes[i*len(cloudsat_scenes)//(64*100):(i)*len(cloudsat_scenes)//(64*100)+len(cloudsat_scenes)//(64*100), :,:, :])
            print(generated_scenes.shape)
            generated_scenes = (torch.transpose(generated_scenes,2,3))
            print(generated_scenes.shape)

            generated_scenes = generated_scenes.reshape(-1, 64)
            generated_scenes = generated_scenes.detach().numpy()
            print(generated_scenes.shape)
            generated_scenes = np.array(generated_scenes)
            print(i)
            if i == 0 :
                all_generated = generated_scenes
            else:
                all_generated = np.append(all_generated,generated_scenes,axis=0)

        #D_gen = [len(cloudsat_scenes)%(64*100), 64, 1]
        #generated_scenes = netG(torch.randn(D_gen), modis_scenes[100*len(cloudsat_scenes)//(64*100):100*len(cloudsat_scenes)//(64*100) + len(cloudsat_scenes)%(64*100), :,:, :])
        #generated_scenes = generated_scenes.view(-1, 64)
        #generated_scenes = generated_scenes.detach().numpy()
        #generated_scenes = np.array(generated_scenes)
        #all_generated = np.append(all_generated,generated_scenes,axis=0)
        all_generated = (all_generated + 1) * (55 / 2) - 35
        print(all_generated.shape)

        name_string = 'generated_scenes_for_histogram'
        filename = 'CGAN_' + name_string + '.h5'
        hf = h5py.File(filename, 'w')
        hf.create_dataset('all_generated', data=all_generated)
        hf.close()

'''
print('Starting to load generated scenes')
location = './'
file_string = location + 'CGAN_generated_scenes_for_histogram' + '.h5'
hf = h5py.File(file_string, 'r')

all_generated = torch.tensor(hf.get('all_generated'))
print('Generated scenes loaded')
'''
total_histogram_generated = np.ones([64, n_bins])

for i in range(0, 64):
    scenes_histogram, bin_edges = np.histogram(all_generated[:, i], bins=n_bins, range=(-35, 20), density=False)
    total_histogram_generated[i] = scenes_histogram
total_sum_generated = np.sum(total_histogram_generated)

total_histogram_generated = total_histogram_generated/(dbz_per_bin*total_sum_generated)


new_total_generated = total_histogram_generated[:, n_removed:n_bins]

x = np.linspace(bin_edges[n_removed], 20, n_bins - n_removed)
y = np.linspace(1, 16.36, 64)
f, ax = plt.subplots(1, 1, figsize=(10.5,15))

pcm = ax.pcolormesh(x, y, new_total_generated)
#ax.set_ylabel("Altitude [km]")
ax.set_xlabel("Reflectivity [dBZ]", fontsize=28)
#ax.set_aspect((n_bins - n_removed) / 64)
ax.set_aspect(135/64)
cb = f.colorbar(pcm, ax=ax, orientation='horizontal', fraction=0.049, pad=0.15, format=ticker.FuncFormatter(fmt))
cb.set_label('Norm. occurrence (Generated) $10^{-4}$ [dBZ$^{-1}$]', fontsize=28)
ax.tick_params(axis='both', which='major', labelsize=26)
cb.ax.tick_params(labelsize=26, rotation=45)
plt.savefig('histogram_generated_CGAN')

difference_histogram = new_total_generated - new_total
x = np.linspace(bin_edges[n_removed], 20, n_bins - n_removed)
y = np.linspace(1, 16.36, 64)
f, ax = plt.subplots(1, 1, figsize=(10.5,15))
pcm = ax.pcolormesh(x, y, difference_histogram, vmin = -0.00021,  vmax = 0.00021,cmap= 'seismic')
#ax3.set_ylabel("Altitude", fontsize=14)
#ax3.set_ylabel("Altitude [km]", fontsize=28)
ax.set_xlabel("Reflectivity [dBZ]", fontsize=28)
#ax.set_aspect((num_bins-num_removed_bins)/64)
ax.set_aspect(135/64)
#ax3.set_aspect('auto')
#ax3.set_title('Occurence difference')
cb=f.colorbar(pcm,ax=ax, orientation = 'horizontal', fraction =0.049, pad=0.15, format=ticker.FuncFormatter(fmt))
cb.set_label('Occurence bias (Real - Gen.) $10^{-4}$ [dBZ$^{-1}$]', fontsize=28)
ax.tick_params(axis='both', which='major', labelsize=26)
#cb3.set_ticks(np.arange(-0.0075, 0.01, step=0.0025))
cb.ax.tick_params(labelsize=26, rotation = 45)
#f3.tight_layout()
plt.savefig('Difference_histogram_CGAN_real_and_generated')
print('done')