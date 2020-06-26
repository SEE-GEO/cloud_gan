import shutil, os
import random
import sys
from os import path
file_name_base = 'rr_modis_cloudsat_data_2015_'
file_location = '/cephyr/users/svcarl/Vera/cloud_gan/gan/temp_transfer/modis_cloudsat_data/'
training_destination = file_location + 'training_data/'
test_destination = file_location + 'test_data/'
num_files = 4900
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
