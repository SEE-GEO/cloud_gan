import os
import shutil
import numpy as np

import numpy.ma as ma
from datetime import datetime
import pandas as pd
import h5py
import torch
import matplotlib.pyplot as plt
from wxdata.index import *
from wxdata import Index

dendrite_path = "~/Dendrite"

index_ecmwf = Index.load(os.path.join(dendrite_path,
                                      "SatData/CloudSat/emwf_aux.index"))
print(index_ecmwf)
index_cloudsat = Index.load(os.path.join(dendrite_path,
                                    "SatData/CloudSat/2b_geoprof.index"))
print(index_cloudsat)
start = datetime(2015, 1, 1)
end = datetime(2016, 1, 1)

files_ecmwf = index_ecmwf.get_files("CloudSat_ECMWF_Aux",
                                    start=start,
                                    end=end)

files_cloudsat = index_cloudsat.get_files("CloudSat_2b_GeoProf",
                                         start=start,
                                       end=end)


files_cloudsat.sort(key= lambda x: x.start_time)
files_ecmwf.sort(key= lambda x: x.start_time)

seen_start_times = set()
new_list = []
for obj in files_ecmwf:
    if obj.start_time not in seen_start_times:
        new_list.append(obj)
        seen_start_times.add(obj.start_time)
files_ecmwf=new_list

new_files_cloudsat = []
keep_index = 0
while keep_index <= len(files_cloudsat)-2:
    sec_between_orbits = (files_cloudsat[keep_index + 1].start_time - files_cloudsat[keep_index].start_time).total_seconds()
    #print('Seconds between orbits cloudsat: ', sec_between_orbits)
    if sec_between_orbits < 5000:
        new_files_cloudsat.append(files_cloudsat[keep_index])
        keep_index = keep_index + 1
    else:
        new_files_cloudsat.append(files_cloudsat[keep_index])
    keep_index = keep_index+1

files_cloudsat = new_files_cloudsat


new_files_ecmwf = []
keep_index = 0
while keep_index <= len(files_ecmwf)-2:
    sec_between_orbits = (files_ecmwf[keep_index + 1].start_time - files_ecmwf[keep_index].start_time).total_seconds()
    #print('Seconds between orbits ecmwf: ',sec_between_orbits)
    if sec_between_orbits < 5000:
        new_files_cloudsat.append(files_ecmwf[keep_index])
        keep_index = keep_index + 1
    else:
        new_files_ecmwf.append(files_ecmwf[keep_index])
    keep_index = keep_index+1

files_ecmwf = new_files_ecmwf

hf = h5py.File('./TemperatureData/ecmwf_sync_list_number4884.h5','r')
disp_cs = np.array(hf.get('displacement_vector_cs'))
disp_ecmwf = np.array(hf.get('displacement_vector_ecmwf'))
min_index = np.array(hf.get('min_index_vector'))
#disp_cs_conc = np.append(disp_cs_conc, disp_cs)
#disp_ecmwf_conc = np.append(disp_ecmwf_conc, disp_ecmwf)
#min_index_conc = np.append(min_index_conc, min_index)

print(disp_ecmwf.shape)
print(disp_cs.shape)
print(min_index.shape)

print(disp_ecmwf[0:10])
print(disp_cs[0:10])

hf = h5py.File('./modis_cloudsat_data/test_data/modis_cloudsat_test_index_list.h5','r')
index_list = np.array(hf.get('index_list'))

print(index_list.shape)

ecmwf_save_list = []
cs_save_list = []
min_index_save_list = []

for ind in index_list:
    for i in range(0,len(disp_cs)):
        if disp_cs[i] == ind:
            cs_save_list = np.append(cs_save_list,disp_cs[i])
            ecmwf_save_list = np.append(ecmwf_save_list,disp_ecmwf[i])
            min_index_save_list = np.append(min_index_save_list,min_index[i])



#File number 783 (at index 80) is the only one in index_list that is missing from disp_cs
print('Save_list shapes')
print(ecmwf_save_list.shape)
print(cs_save_list.shape)

print(cs_save_list[75:85])
print(index_list[75:85])

hf = h5py.File('Temperature_index_list.h5', 'w')
hf.create_dataset('ecmwf_list', data=ecmwf_save_list)
hf.create_dataset('cs_list', data=cs_save_list)
hf.create_dataset('min_index_list', data=min_index_save_list)
hf.close()


f, ax = plt.subplots(1,1)
#ax.plot(disp_ecmwf, color='blue')
#ax.plot(disp_cs, color='red')
ax.plot(index_list, color='black')
ax.plot(cs_save_list, color='green')
ax.plot(ecmwf_save_list, color='red')

plt.show()
'''

for ind_cs in cs_save_list: 
    ind_cs = int(ind_cs)
    file_cloudsat = files_cloudsat[ind_cs].open()
    latitude_cs = file_cloudsat.latitude




k = 0
for ind_ecmwf in ecmwf_save_list:
    ind_ecmwf = int(ind_ecmwf)
    file_ecmwf = files_ecmwf[ind_ecmwf].open()
    temperatures = file_ecmwf.ecmwf_temperature
    latitude_ecmwf = file_ecmwf.latitude
    longitude_ecmwf = file_ecmwf.longitude
    altitude_ecmwf = file_ecmwf.EC_height
    min_value = min_index[k]
    k = k+1
    min_value = int(min_value)
    print(min_value)
    print('Shapes of loaded parameters before min_index cutoff')
    print(temperatures.shape)
    print(latitude_ecmwf.shape)
    print(longitude_ecmwf.shape)
    print(altitude_ecmwf.shape)

    temperatures = temperatures[min_value:]
    latitude_ecmwf = latitude_ecmwf[min_value:]
    longitude_ecmwf = longitude_ecmwf[min_value:]
    print('New shapes of loaded parameters')
    print(temperatures.shape)
    print(latitude_ecmwf.shape)
    print(longitude_ecmwf.shape)
    print(altitude_ecmwf.shape)

    temperatures = temperatures[0:len(temperatures)-len(temperatures)%64,:]
    temperatures = temperatures.reshape(-1,64,125)
    latitude_ecmwf = latitude_ecmwf[0:len(latitude_ecmwf)-len(latitude_ecmwf)%64]
    latitude_ecmwf = latitude_ecmwf.reshape(-1,64)
    longitude_ecmwf = longitude_ecmwf[0:len(longitude_ecmwf)-len(longitude_ecmwf)%64]
    longitude_ecmwf = longitude_ecmwf.reshape(-1,64)

'''
