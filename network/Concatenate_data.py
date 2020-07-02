from os import path
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime

time_start = datetime.datetime.now()
print('Script started',time_start)
for cloudsat_file in range(0 ,4900):
    location = './modis_cloudsat_data/training_data/'
    file_string = location + 'rr_modis_cloudsat_data_2015_' + str(cloudsat_file).zfill(4) +'.h5'
    if path.exists(file_string):
        hf = h5py.File(file_string, 'r')

        cloudsat_scenes_temp = torch.tensor(hf.get('rr')).view(-1, 1, 64, 64)
        modis_scenes_temp = torch.tensor(hf.get('emissivity')).view(-1, 1, 64, 10).float()
        modis_scenes_temp[modis_scenes_temp == -40] = 0
        if cloudsat_file == 0:
            cloudsat_scenes = cloudsat_scenes_temp
            modis_scenes = modis_scenes_temp
        else:
            cloudsat_scenes = torch.cat([cloudsat_scenes, cloudsat_scenes_temp], 0)
            modis_scenes = torch.cat([modis_scenes, modis_scenes_temp], 0)
        if cloudsat_file%100==0:
            print(cloudsat_file)
time_files_loaded = datetime.datetime.now()
print('Files loaded after', (time_files_loaded - time_start).total_seconds(), 'seconds')
print('Number of scenes ', len(modis_scenes))
minimum = torch.tensor(np.zeros([10,1]))
maximum = torch.tensor(np.zeros([10,1]))
f,ax = plt.subplots(2,1)
for i in range(0,10):
    minimum[i] = torch.min(modis_scenes[:,0,:,i]).item()
    if minimum[i] < 0:
        print('min',str(minimum[i]), 'band number ', i)
    modis_scenes[:, 0, :, i] -= minimum[i]
    maximum[i] = torch.max(modis_scenes[:,0,:,i]).item()
    modis_scenes[:, 0, :, i] /= maximum[i]/2
    modis_scenes[:, 0, :, i] -=1
    #print('max',str(maximum[i]))

'''
nbins=50
for i in range(0,2):
    #print((np.histogram(modis_scenes[:, 0, :, i], bins=nbins, range=(-1, 1), density=True)[0]).shape)
    ax[i].hist(modis_scenes[:, 0, :, i*4], bins=nbins, range=(-1, 1), density=True)
plt.savefig('distr_modis_channel_modified_')
'''

time_end = datetime.datetime.now()
total_time = (time_end - time_start).total_seconds()
print('Script finished', time_end)
print('Total time', total_time)

name_string = 'training_data_conc'
filename = 'modi_cloudsat_' + name_string + '.h5'
hf = h5py.File(filename, 'w')
hf.create_dataset('cloudsat_scenes', data=cloudsat_scenes)
hf.create_dataset('modis_scenes', data=modis_scenes)
hf.create_dataset('max', data=maximum)
hf.create_dataset('min', data=minimum)
hf.close()