from os import path
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime

time_start = datetime.datetime.now()
print('Script started',time_start)
counter = 0

hf = h5py.File('Temperature_index_list.h5','r')
cs_list = np.array(hf.get('cs_list'))

for cloudsat_file in cs_list:
    cloudsat_file = int(cloudsat_file)
    location = './TemperatureData/'
    file_string = location + 'CGAN_testset_with_temperature_2015_' + str(cloudsat_file).zfill(4) +'.h5'
    if path.exists(file_string):
        hf = h5py.File(file_string, 'r')

        cloudsat_scenes_temp = torch.tensor(hf.get('rr')).view(-1, 1, 64, 64)
        modis_scenes_temp_temp = np.array(torch.tensor(hf.get('emissivity')).view(-1, 1, 64, 10).float())
        mask = modis_scenes_temp_temp == -40
        modis_scenes_temp_temp[mask]=-1
        modis_scenes_temp = np.ma.array(modis_scenes_temp_temp,mask=mask, fill_value=-1)

        latitude_temp = torch.tensor(hf.get('latitude')).view(-1, 64)
        longitude_temp = torch.tensor(hf.get('longitude')).view(-1, 64)
        elevation_temp = torch.tensor(hf.get('DEM_elevation')).view(-1, 64)

        temperature_temp = torch.tensor(hf.get('temperatures')).view(-1,1,64,125)
        altitude_ecmwf_temp = torch.tensor(hf.get('altitude_ecmwf')).view(-1, 125)


        if counter == 0:
            counter=1
            cloudsat_scenes = cloudsat_scenes_temp
            modis_scenes = modis_scenes_temp
            latitude = latitude_temp
            longitude = longitude_temp
            elevation = elevation_temp
            temperature = temperature_temp
            altitude_ecmwf = altitude_ecmwf_temp
        else:
            cloudsat_scenes = torch.cat([cloudsat_scenes, cloudsat_scenes_temp], 0)
            modis_scenes = np.ma.concatenate([modis_scenes, modis_scenes_temp], 0)
            latitude = torch.cat([latitude, latitude_temp], 0)
            longitude = torch.cat([longitude, longitude_temp], 0)
            elevation = torch.cat([elevation, elevation_temp], 0)
            temperature = torch.cat([temperature,temperature_temp],0)
            altitude_ecmwf = torch.cat([altitude_ecmwf,altitude_ecmwf_temp],0)
        if cloudsat_file%10==0:
            print(cloudsat_file)
time_files_loaded = datetime.datetime.now()
print('Files loaded after', (time_files_loaded - time_start).total_seconds(), 'seconds')
print('Number of scenes ', len(modis_scenes))
minimum = (np.zeros([10,1]))
maximum = (np.zeros([10,1]))
file_string = './modis_cloudsat_data/' + 'modis_cloudsat_training_data_conc' + '.h5'
hf = h5py.File(file_string, 'r')
minimum= torch.tensor(hf.get('min'))
maximum= torch.tensor(hf.get('max'))
f,ax = plt.subplots(2,1)
print(modis_scenes.shape)
for i in range(0,10):
    #minimum[i] = (modis_scenes[:,0,:,i]).min()


    print('min',str(minimum[i]), 'band number ', i)
    modis_scenes[:, 0, :, i] -= minimum[i] - 0.0001
    #maximum[i] = np.ma.max(modis_scenes[:,0,:,i])
    print('max', str(maximum[i]), 'band number ', i)
    modis_scenes[:, 0, :, i] /= maximum[i]/2
    modis_scenes[:, 0, :, i] -=1


    #print('max',str(maximum[i]))
np.ma.set_fill_value(modis_scenes, -1)

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

name_string = 'CGAN_test_data_with_temp_conc'
filename = name_string + '.h5'
hf = h5py.File(filename, 'w')
hf.create_dataset('cloudsat_scenes', data=cloudsat_scenes)
hf.create_dataset('modis_scenes', data=modis_scenes)
hf.create_dataset('latitude', data=latitude)
hf.create_dataset('longitude', data=longitude)
hf.create_dataset('DEM_elevation', data=elevation)
hf.create_dataset('temperature', data=temperature)
hf.create_dataset('altitude_ecmwf', data=altitude_ecmwf)
hf.create_dataset('max', data=maximum)
hf.create_dataset('min', data=minimum)
hf.close()