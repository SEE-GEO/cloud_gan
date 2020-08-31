import h5py
import shutil, os
import random
import sys
import numpy as np
from os import path
file_name_base = 'rr_data_latitude_longitude_elevation2015_'
file_location = './GAN_elevation/'
training_destination = file_location + 'training_data/'
test_destination = file_location + 'test_data/'
num_files = 4900

hf = h5py.File('./rr_data/index_list.h5', 'r')
index_list = (np.array(hf.get('index_list')))
for ind in index_list:
    file_ending = str(ind).zfill(4) + '.h5'
    file_name = file_name_base + file_ending
    if path.exists(file_location + file_name):
        shutil.move(file_location + file_name, test_destination)


'''
quota = 0.9 # numberoftraining/numberoftest
for i in range(0,num_files):
    file_ending = str(i).zfill(4) + '.h5'
    file_name=file_name_base + file_ending
    if path.exists(file_location + file_name):
        r = random.random()
        if r < quota:
            shutil.move(file_location + file_name, training_destination)
        else:
            shutil.move(file_location + file_name, test_destination)
'''