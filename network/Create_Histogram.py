
import h5py
import torch
import  numpy as np
import matplotlib.pyplot as plt
from os import path
#from mpl_toolkits.axes_grid1 import make_axes_locatable

#Create one file with all cloudsat_scenes
from GAN_generator import GAN_generator

counter = 0
for cloudsat_file in range(0,5000):
    location = './rr_data/test_data/'
    file_string = location + 'rr_data_2015_' + str(cloudsat_file).zfill(4) +'.h5'
    if path.exists(file_string):
        hf = h5py.File(file_string, 'r')

        cloudsat_scenes_temp = torch.tensor(hf.get('rr')).view(-1,64,64)
        cloudsat_scenes_temp = (cloudsat_scenes_temp + 1)*(55 / 2)-35
        counter +=1
        print(counter)

        if counter == 1:
            cloudsat_scenes=cloudsat_scenes_temp
        else:
            cloudsat_scenes = torch.cat([cloudsat_scenes,cloudsat_scenes_temp],0)
       # cloudsat_scenes = cloudsat_scenes + cloudsat_scenes_temp
print(cloudsat_scenes.shape)
print(cloudsat_scenes[0][0][0].shape) #Reflectivities at different altitudes for one position
cloudsat_scenes = cloudsat_scenes.view(-1,64)
#cloudsat_scenes = np.array(cloudsat_scenes)
print('Shape of cloudsat_scenes', cloudsat_scenes.shape)

total_histogram = np.ones([64,165])
cloudsat_at_altitude = cloudsat_scenes[:,0]
print(cloudsat_at_altitude.shape)
for i in range(0,64):
    scenes_histogram, bin_edges = np.histogram(cloudsat_scenes[:,i], bins=165, range=(-35,20), density=True)
    total_histogram[i] = scenes_histogram
print(total_histogram.shape)
#print(bin_edges)
new_total = total_histogram[:,30:165]
print(new_total.shape)


x=np.linspace(bin_edges[30],20,135)
y=np.linspace(1, 18.92,64)
f,ax = plt.subplots(1, 1)
pcm = ax.pcolormesh(x, y, new_total)
ax.set_ylabel("Altitude [km]")
ax.set_xlabel("Reflectivity [dBZ]")
ax.set_aspect(135/64)
#divider = make_axes_locatable(ax)
#cax = divider.append_axes('bottom', size='100%', pad=0.05)
#cax = f.add_axes([0.09, 0.06, 0.84, 0.02])
cb=f.colorbar(pcm,ax=ax , orientation = 'horizontal', fraction = 0.049, pad=0.15)
cb.set_label('Norm. occurrence (real)')
plt.savefig('histogram_gan_real')

#D_gen = [50, 64, 6]
D_gen = [len(cloudsat_scenes)//(64*100), 64, 6]
H_gen=[384,16384, 256, 128, 64, 1]
netG = GAN_generator(H_gen)
folder = '/cephyr/users/svcarl/Vera/cloud_gan/gan/temp_transfer/training_results_gan_ver3/'
if path.exists(folder + 'network_parameters.pt'):
    print('adasdefavcsxasxa')
    with torch.no_grad():
        folder_path = '/cephyr/users/svcarl/Vera/cloud_gan/gan/temp_transfer/training_results_gan_ver3/'
        checkpoint = torch.load(folder_path + 'network_parameters.pt', map_location=torch.device('cpu'))
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
        total_histogram_generated = np.ones([64,165])
        print(all_generated.shape)
        for i in range(0, 64):
            scenes_histogram, bin_edges = np.histogram(all_generated[:, i], bins=165, range=(-35, 20), density=True)
            total_histogram_generated[i] = scenes_histogram

        new_total_generated = total_histogram_generated[:,30:165]

        x = np.linspace(bin_edges[30], 20, 135)
        y=np.linspace(1, 18.92,64)
        f, ax = plt.subplots(1, 1)
        pcm = ax.pcolormesh(x, y, new_total_generated)
        ax.set_ylabel("Altitude [km]")
        ax.set_xlabel("Reflectivity [dBZ]")
        ax.set_aspect(135 / 64)
        cb=f.colorbar(pcm,ax=ax , orientation = 'horizontal', fraction = 0.049, pad=0.15)
        cb.set_label('Norm. occurrence (generated)')
        plt.savefig('histogram_gan_generated_ver3')

    difference_histogram = new_total_generated - new_total
    x = np.linspace(bin_edges[30], 20, 135)
    y = np.linspace(1, 18.92, 64)
    f, ax = plt.subplots(1, 1)
    pcm = ax.pcolormesh(x, y, difference_histogram, cmap= 'seismic')
    ax.set_ylabel("Altitude [km]")
    ax.set_xlabel("Reflectivity [dBZ]")
    ax.set_aspect(135/64)

    cb=f.colorbar(pcm,ax=ax , orientation = 'horizontal', fraction = 0.049, pad=0.15)
    cb.set_label('Occurence bias (gen. - real)')
    plt.savefig('histogram_difference_ver3')


