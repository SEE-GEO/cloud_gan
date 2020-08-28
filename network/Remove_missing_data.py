import h5py
from IceWaterPathMethod import IceWaterPathMethod
from GAN_generator import GAN_generator
from plot_cloud import plot_cloud
from os import path
import torch
import numpy as np
import matplotlib.pyplot as plt
folder = './'


location = './modis_cloudsat_data/'
file_string = location + 'modis_cloudsat_test_data_conc' + '.h5'
hf = h5py.File(file_string, 'r')

cloudsat_scenes = torch.tensor(hf.get('cloudsat_scenes'))

modis_scenes = torch.tensor(hf.get('modis_scenes'))
maximum = torch.tensor(hf.get('max'))
minimum = torch.tensor(hf.get('min'))


print(cloudsat_scenes.shape)
print(modis_scenes.shape)
cloudsat_scenes_temp=[]
modis_scenes_temp = []
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
cloudsat_scenes=cloudsat_scenes[index_list,:,:,:]
modis_scenes=modis_scenes[index_list,:,:,:]
name_string = 'test_data_conc_ver2'
filename = 'modis_cloudsat_' + name_string + '.h5'
hf = h5py.File(filename, 'w')
hf.create_dataset('cloudsat_scenes', data=cloudsat_scenes)
hf.create_dataset('modis_scenes', data=modis_scenes)
hf.create_dataset('max', data=maximum)
hf.create_dataset('min', data=minimum)
hf.close()