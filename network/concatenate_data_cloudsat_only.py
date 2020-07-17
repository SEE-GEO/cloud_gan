from os import path
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime

time_start = datetime.datetime.now()
print('Script started',time_start)
counter = 0
for cloudsat_file in range(0 ,5000):
    location = './rr_data/test_data/'
    file_string = location + 'rr_data_2015_' + str(cloudsat_file).zfill(4) +'.h5'
    if path.exists(file_string):
        hf = h5py.File(file_string, 'r')

        cloudsat_scenes_temp = torch.tensor(hf.get('rr')).view(-1, 1, 64, 64)

        if counter == 0:
            counter=1
            cloudsat_scenes = cloudsat_scenes_temp
        else:
            cloudsat_scenes = torch.cat([cloudsat_scenes, cloudsat_scenes_temp], 0)
        if cloudsat_file%100==0:
            print(cloudsat_file)
time_files_loaded = datetime.datetime.now()
print('Files loaded after', (time_files_loaded - time_start).total_seconds(), 'seconds')

name_string = 'test_data_conc'
filename = 'cloudsat_' + name_string + '.h5'
hf = h5py.File(filename, 'w')
hf.create_dataset('cloudsat_scenes', data=cloudsat_scenes)
hf.close()