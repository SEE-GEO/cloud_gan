from os import path
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime





folder1 ='./rr_modis_cloudsat_data/'
file_name1 = 'modis_cloudsat_ElevLatLong_test_data_conc_ver2.h5'
file_string1 = folder1 + file_name1


folder2 ='./rr_modis_cloudsat_data/'
file_name2 = 'modis_cloudsat_test_data_conc_ver2.h5'
file_string2 = folder2 + file_name2
cloudsat_scenes_0 = []
cloudsat_scenes_20 = []
cloudsat_scenes_40 = []
cloudsat_scenes_60 = []
modis_scenes_0 = []
modis_scenes_20 = []
modis_scenes_30 = []
modis_scenes_40= []
if path.exists(file_string1) and path.exists(file_string2):
        hf1 = h5py.File(file_string1, 'r')

        cloudsat_scenes_temp = (hf1.get('cloudsat_scenes'))
        modis_scenes_temp = np.array(torch.tensor(hf1.get('modis_scenes')).view(-1, 1, 64, 10).float())
        latitude = np.array(hf1.get('latitude'))
        print(cloudsat_scenes_temp.shape)
        print(modis_scenes_temp.shape)
        print(latitude.shape)

        for i in range(0,len(latitude)):
                current_latitude = np.abs(latitude[i,0])
                if current_latitude>0 and current_latitude<20:
                        if len(cloudsat_scenes_0) == 0:
                                cloudsat_scenes_0 = cloudsat_scenes_temp[i]
                                modis_scenes_0 = modis_scenes_temp[i]
                        else:
                                cloudsat_scenes_0 = np.append(cloudsat_scenes_0,cloudsat_scenes_temp[i],0)
                                modis_scenes_0 = np.append(modis_scenes_0,modis_scenes_temp[i],0)
        
                elif current_latitude>20 and current_latitude<40:
                        if len(cloudsat_scenes_20) == 0:
                                cloudsat_scenes_20 = cloudsat_scenes_temp[i]
                                modis_scenes_20 = modis_scenes_temp[i]
                        else:
                                cloudsat_scenes_20 = np.append(cloudsat_scenes_20,cloudsat_scenes_temp[i],0)
                                modis_scenes_20 = np.append(modis_scenes_20,modis_scenes_temp[i],0)
                elif current_latitude > 40 and current_latitude < 60:
                        if len(cloudsat_scenes_40) == 0:
                                cloudsat_scenes_40 = cloudsat_scenes_temp[i]
                                modis_scenes_40 = modis_scenes_temp[i]
                        else:
                                cloudsat_scenes_40 = np.append(cloudsat_scenes_40,cloudsat_scenes_temp[i],0)
                                modis_scenes_40 = np.append(modis_scenes_40,modis_scenes_temp[i],0)
                elif current_latitude > 60:
                        if len(cloudsat_scenes_60) == 0:
                                cloudsat_scenes_60 = cloudsat_scenes_temp[i]
                                modis_scenes_60 = modis_scenes_temp[i]
                        else:
                                cloudsat_scenes_60 = np.append(cloudsat_scenes_60,cloudsat_scenes_temp[i],0)
                                modis_scenes_60 = np.append(modis_scenes_60,modis_scenes_temp[i],0)

        base_file_name = 'modis_cloudsat_latitude_'
        ending = ['0','20','40','60']
        for i in range(0,len(ending)):
                file_name = base_file_name + ending[i] + '.h5'
                hf = h5py.File(file_name, 'w')
                if i ==0:
                        print('modis_0 ',modis_scenes_0.shape)
                        print('cloudsat_0',cloudsat_scenes_0.shape)
                        hf.create_dataset('modis_scenes', data=modis_scenes_0)
                        hf.create_dataset('cloudsat_scenes', data=cloudsat_scenes_0)

                if i == 1:

                        print('modis_20 ',modis_scenes_20.shape)
                        print('cloudsat_20',cloudsat_scenes_20.shape)
                        hf.create_dataset('modis_scenes', data=modis_scenes_20)
                        hf.create_dataset('cloudsat_scenes', data=cloudsat_scenes_20)

                if i == 2:

                        print('modis_40 ',modis_scenes_40.shape)
                        print('cloudsat_40',cloudsat_scenes_40.shape)
                        hf.create_dataset('modis_scenes', data=modis_scenes_40)
                        hf.create_dataset('cloudsat_scenes', data=cloudsat_scenes_40)

                if i == 3:

                        print('modis_60 ',modis_scenes_60.shape)
                        print('cloudsat_60',cloudsat_scenes_60.shape)
                        hf.create_dataset('modis_scenes', data=modis_scenes_60)
                        hf.create_dataset('cloudsat_scenes', data=cloudsat_scenes_60)

                hf.close()
