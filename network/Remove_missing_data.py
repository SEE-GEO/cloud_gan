import h5py
from os import path
import torch
import numpy as np
import matplotlib.pyplot as plt
folder = './'

location = './modis_cloudsat_data/2016/'
file_string = location + 'modis_cloudsat_test_data_2016_normed2015_conc' + '.h5'
#hf = h5py.File('CGAN_test_data_with_temp_conc.h5', 'r')
hf = h5py.File(file_string,'r')

#hf2 = h5py.File('./CGAN_elevation/CGAN_elev_lat_long_test_data_conc.h5','r')
cloudsat_scenes = torch.tensor(hf.get('cloudsat_scenes'))
modis_scenes = torch.tensor(hf.get('modis_scenes'))
maximum = torch.tensor(hf.get('max'))
minimum = torch.tensor(hf.get('min'))
latitude = torch.tensor(hf.get('latitude'))
longitude = torch.tensor(hf.get('longitude'))
elevation = torch.tensor(hf.get('DEM_elevation'))
#temperature = torch.tensor(hf.get('temperature'))
#altitude_ecmwf = torch.tensor(hf.get('altitude_ecmwf'))

#print(altitude_ecmwf.shape)
#print(altitude_ecmwf[0])
#print(temperature.shape)

# For taking out temperature bins below:
'''
k=0
bottom_height_index = []
for i in range(0,len(altitude_ecmwf)):
        while altitude_ecmwf[i,k] > 1000:
            k = k+1
        bottom_height_index = np.append(bottom_height_index,k)

temperature_new = temperature[:,:,:,k-64:k]
altitude_ecmwf = altitude_ecmwf[:,k-64:k]

print(temperature_new.shape)
'''

print(cloudsat_scenes.shape)
print(modis_scenes.shape)
print(latitude.shape)
index_list=[]
counter_removed=0
for i in range(0,len(modis_scenes)):
    count = 0
    for j in range(0,64):
        if modis_scenes[i,0,j,0] == -1:
            count=count+1
    if count > 21:
        counter_removed = counter_removed + 1
        print('bad scene ',count,', number of removed scenes: ',counter_removed)
    else:
        index_list = np.append(index_list, i)

print(index_list)
print('Shape of the different fields before removal')
print(latitude.shape)
print(longitude.shape)
print(elevation.shape)
print(cloudsat_scenes.shape)
print(modis_scenes.shape)
#print(temperature_new.shape)
cloudsat_scenes=cloudsat_scenes[index_list,:,:,:]
modis_scenes=modis_scenes[index_list,:,:,:]
latitude = latitude[index_list,:]
longitude = longitude[index_list,:]
elevation = elevation[index_list,:]
#temperature_new = temperature_new[index_list,:,:,:]
#altitude_ecmwf = altitude_ecmwf[0]

print('Shape of the different fields after removal')
print(latitude.shape)
print(longitude.shape)
print(elevation.shape)
print(cloudsat_scenes.shape)
print(modis_scenes.shape)
#print(temperature_new.shape)

name_string = 'test_data_conc_2016_normed2015_ver2'
filename = 'modis_cloudsat_ElevLatLong_' + name_string + '.h5'
#hf = h5py.File('CGAN_test_data_with_temp_conc_ver2.h5', 'w')
hf = h5py.File(filename, 'w')
hf.create_dataset('cloudsat_scenes', data=cloudsat_scenes)
hf.create_dataset('modis_scenes', data=modis_scenes)
hf.create_dataset('max', data=maximum)
hf.create_dataset('min', data=minimum)
hf.create_dataset('latitude', data=latitude)
hf.create_dataset('longitude', data=longitude)
hf.create_dataset('DEM_elevation', data=elevation)
#hf.create_dataset('temperature', data=temperature_new)
#hf.create_dataset('altitude_ecmwf', data=altitude_ecmwf)
hf.create_dataset('index_list', data = index_list)
hf.close()

