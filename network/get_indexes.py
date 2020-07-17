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
        if counter == 0:
            index_list = [cloudsat_file]
            counter = 1
        else:
            index_list = np.append(index_list,cloudsat_file)



name_string = 'index_list'
filename = name_string + '.h5'
hf = h5py.File(filename, 'w')
hf.create_dataset('index_list', data=index_list)
hf.close()