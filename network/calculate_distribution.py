from os import path
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt

for cloudsat_file in range(0, 10):
    location = './modis_cloudsat_data/training_data/'
    file_string = location + 'rr_modis_cloudsat_data_2015_' + str(cloudsat_file).zfill(4) + '.h5'
    if path.exists(file_string):
        hf = h5py.File(file_string, 'r')

        #cloudsat_scenes_temp = torch.tensor(hf.get('rr')).view(-1, 1, 64, 64)
        modis_scenes_temp = torch.tensor(hf.get('emissivity')).view(-1, 1, 64, 10).float()
        modis_scenes_temp[modis_scenes_temp == -40] = 0
        if cloudsat_file == 0:
            #cloudsat_scenes = cloudsat_scenes_temp
            modis_scenes = modis_scenes_temp
        else:
            #cloudsat_scenes = torch.cat([cloudsat_scenes, cloudsat_scenes_temp], 0)
            modis_scenes = torch.cat([modis_scenes, modis_scenes_temp], 0)
nbins = 100
print(modis_scenes.shape)
modis_scenes = np.array(modis_scenes)
modis_histogram = np.zeros([100,64,10])
f,ax = plt.subplots(10,1)
plot=np.zeros([10,100])
for j in range (0,10):
    print((np.histogram(modis_scenes[:,0,:,j], bins=nbins, range = (-1,5),density=True)[0]).shape)
    ax[j].hist(modis_scenes[:,0,:,j], bins=nbins, range = (-1,10),density=True)
    plt.savefig('distr_modis_channel_' + str(j))