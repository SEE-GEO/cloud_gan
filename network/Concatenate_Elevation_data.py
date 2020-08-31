from os import path
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime

time_start = datetime.datetime.now()
print('Script started',time_start)

counter = 0
for file_number in range(0 ,5000):
    location = './CGAN_elevation/test_data/'
    file_string = location + 'data_latitudes_longitudes2015_' + str(file_number).zfill(4) +'.h5'
    if path.exists(file_string):
        hf = h5py.File(file_string, 'r')
        latitude_temp = torch.tensor(hf.get('latitude')).view(-1, 64)
        longitude_temp = torch.tensor(hf.get('longitude')).view(-1, 64)
        elevation_temp = torch.tensor(hf.get('DEM_elevation')).view(-1, 64)

        if counter == 0:
            counter=1
            latitude = latitude_temp
            longitude = longitude_temp
            elevation = elevation_temp
        else:
            latitude = torch.cat([latitude, latitude_temp], 0)
            longitude = torch.cat([longitude, longitude_temp], 0)
            elevation = torch.cat([elevation, elevation_temp], 0)

        if file_number % 100 == 0:
            print('File number ',file_number)
time_files_loaded = datetime.datetime.now()
print('Files loaded after', (time_files_loaded - time_start).total_seconds(), 'seconds')
print('Number of scenes ', len(latitude))
print(latitude.shape)


name_string = 'test_data_conc'
filename = 'CGAN_elev_lat_long_' + name_string + '.h5'
hf = h5py.File(filename, 'w')
hf.create_dataset('latitude', data=latitude)
hf.create_dataset('longitude', data=longitude)
hf.create_dataset('DEM_elevation', data=elevation)
hf.close()
