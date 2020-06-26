
import os
import os.path
from os import path
import h5py
import numpy.ma as ma
from datetime import datetime
import numpy as np
import torch

import matplotlib.pyplot as plt
from GAN_generator import GAN_generator
from GAN_discriminator import GAN_discriminator
#from Create_Dataset import create_dataset

for cloudsat_file in range(0, 4900):
    location = './modis_cloudsat_data/'
    file_string = location + 'rr_modis_cloudsat_data_2015_' + str(cloudsat_file).zfill(4) + '.h5'
    if path.exists(file_string):
        hf = h5py.File(file_string, 'r')
        print(cloudsat_file)
        cloudsat_scenes_temp = torch.tensor(hf.get('rr')).view(-1, 1, 64, 64)
        if cloudsat_file == 0:
            cloudsat_scenes = cloudsat_scenes_temp
        else:
            cloudsat_scenes = torch.cat([cloudsat_scenes, cloudsat_scenes_temp], 0)

print(len(cloudsat_scenes))
