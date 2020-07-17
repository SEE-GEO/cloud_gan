
import h5py
import torch
import  numpy as np
import matplotlib.pyplot as plt
from os import path
#from mpl_toolkits.axes_grid1 import make_axes_locatable

#Create one file with all cloudsat_scenes
from GAN_generator import GAN_generator
n_bins = 55*3
n_removed = int((30/165)*n_bins)

location = './rr_data/'
file_string = location + 'cloudsat_test_data_conc' + '.h5'
hf = h5py.File(file_string, 'r')

cloudsat_scenes = torch.tensor(hf.get('cloudsat_scenes'))
cloudsat_scenes = (cloudsat_scenes+1)*(55/2)-35
cloudsat_scenes = cloudsat_scenes.view(-1,64)
print('Shape of cloudsat_scenes', cloudsat_scenes.shape)

total_histogram = np.ones([64,n_bins])
cloudsat_at_altitude = cloudsat_scenes[:,0]
print(cloudsat_at_altitude.shape)
for i in range(0,64):
    scenes_histogram, bin_edges = np.histogram(cloudsat_scenes[:,i], bins=n_bins, range=(-34,20), density=True)
    total_histogram[i] = scenes_histogram
print(total_histogram.shape)
#print(bin_edges)
new_total = total_histogram[:,n_removed:n_bins]
print(new_total.shape)


x=np.linspace(bin_edges[n_removed],20,n_bins - n_removed)
y=np.linspace(1, 16.36,64)
f,ax = plt.subplots(1, 1)
pcm = ax.pcolormesh(x, y, new_total)
ax.set_ylabel("Altitude [km]")
ax.set_xlabel("Reflectivity [dBZ]")
ax.set_aspect((n_bins - n_removed)/64)
#divider = make_axes_locatable(ax)
#cax = divider.append_axes('bottom', size='100%', pad=0.05)
#cax = f.add_axes([0.09, 0.06, 0.84, 0.02])
cb=f.colorbar(pcm,ax=ax , orientation = 'horizontal', fraction = 0.049, pad=0.15)
cb.set_label('Norm. occurrence (real)')
plt.savefig('histogram_gan_real_test')

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
            generated_scenes = generated_scenes.view(-1, 64)
            generated_scenes = generated_scenes.detach().numpy()
            generated_scenes = np.array(generated_scenes)
            print(i)
            if i == 0 :
                all_generated = generated_scenes
            else:
                all_generated = np.append(all_generated,generated_scenes,axis=0)

        all_generated = (all_generated + 1) * (55 / 2) - 35
        
                total_histogram_generated = np.ones([64,n_bins])

        for i in range(0, 64):
            scenes_histogram, bin_edges = np.histogram(all_generated[:, i], bins=n_bins, range=(-34, 20), density=True)
            total_histogram_generated[i] = scenes_histogram

        new_total_generated = total_histogram_generated[:,n_removed:n_bins]

        x = np.linspace(bin_edges[n_removed], 20, n_bins - n_removed)
        y=np.linspace(1, 16.36,64)
        f, ax = plt.subplots(1, 1)
        pcm = ax.pcolormesh(x, y, new_total_generated)
        ax.set_ylabel("Altitude [km]")
        ax.set_xlabel("Reflectivity [dBZ]")
        ax.set_aspect((n_bins - n_removed)/64)
        cb=f.colorbar(pcm,ax=ax , orientation = 'horizontal', fraction = 0.049, pad=0.15)
        cb.set_label('Norm. occurrence (generated)')
        plt.savefig('histogram_gan_generated_ver4_test')

    difference_histogram = new_total_generated - new_total
    x = np.linspace(bin_edges[n_removed], 20, n_bins - n_removed)
    y = np.linspace(1, 16.36, 64)
    f, ax = plt.subplots(1, 1)
    pcm = ax.pcolormesh(x, y, difference_histogram, cmap= 'seismic')
    ax.set_ylabel("Altitude [km]")
    ax.set_xlabel("Reflectivity [dBZ]")
    ax.set_aspect((n_bins - n_removed)/64)
    

    cb=f.colorbar(pcm,ax=ax , orientation = 'horizontal', fraction = 0.049, pad=0.15)
    cb.set_label('Occurence bias (gen. - real)')
    plt.savefig('histogram_difference_ver4_test')

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
    scenes_histogram, bin_edges = np.histogram(all_generated[:, i], bins=n_bins, range=(-34, 20), density=True)
    total_histogram_generated[i] = scenes_histogram

new_total_generated = total_histogram_generated[:,n_removed:n_bins]

x = np.linspace(bin_edges[n_removed], 20, n_bins - n_removed)
y=np.linspace(1, 16.36,64)
f, ax = plt.subplots(1, 1)
pcm = ax.pcolormesh(x, y, new_total_generated)
ax.set_ylabel("Altitude [km]")
ax.set_xlabel("Reflectivity [dBZ]")
ax.set_aspect((n_bins - n_removed)/64)
cb=f.colorbar(pcm,ax=ax , orientation = 'horizontal', fraction = 0.049, pad=0.15)
cb.set_label('Norm. occurrence (generated)')
plt.savefig('histogram_gan_generated_ver4_test')

difference_histogram = new_total_generated - new_total
x = np.linspace(bin_edges[n_removed], 20, n_bins - n_removed)
y = np.linspace(1, 16.36, 64)
f, ax = plt.subplots(1, 1)
pcm = ax.pcolormesh(x, y, difference_histogram, cmap= 'seismic')
ax.set_ylabel("Altitude [km]")
ax.set_xlabel("Reflectivity [dBZ]")
ax.set_aspect((n_bins - n_removed)/64)

cb=f.colorbar(pcm,ax=ax , orientation = 'horizontal', fraction = 0.049, pad=0.15)
cb.set_label('Occurence bias (gen. - real)')
plt.savefig('histogram_difference_ver4_test')




