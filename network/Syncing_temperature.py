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
#from Create_Dataset_To_File import create_dataset_to_file
#from Create_Modis_Dataset_To_File import create_dataset_modis

#create_dataset_modis(True)


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


#print('difference')
'''
for i in range(0,4000):
    date_ecmwf1 = files_ecmwf[i].start_time
    date_cloudsat1 = files_cloudsat[i].start_time
    date_ecmwf2 = files_ecmwf[i+1].start_time
    date_cloudsat2 = files_cloudsat[i+1].start_time
    print((date_cloudsat1-date_cloudsat2).total_seconds()/60,(date_ecmwf1-date_ecmwf2).total_seconds()/60)
'''

hf = h5py.File('ecmwf_sync_list_number4800.h5', 'r')
displacement_vector_cs = np.array(hf.get('displacement_vector_cs'))
displacement_vector_ecmwf = np.array(hf.get('displacement_vector_ecmwf'))
min_index_vector = np.array(hf.get('min_index_vector'))
time_displacement = np.array(hf.get('time_displacement'))
#time_displacement = time_displacement[0]
#print(time_displacement, 'time disp')
temp_displacement = np.array(hf.get('temp_displacement'))
#temp_displacement = temp_displacement[0]
#print(displacement_vector_cs.shape, 'loaded disp cs')
'''
time_displacement = 0
temp_displacement = 0
displacement_vector_cs = []
displacement_vector_ecmwf = []
min_index_vector = []
'''

time_for_one_rotation = (files_cloudsat[1].start_time - files_cloudsat[0].start_time).total_seconds()
print('Time for one rotation: ',time_for_one_rotation)

for ind in range(4801, len(files_cloudsat)-1):
#for ind in range(395,402):

    print(' -- NEW LAP! Index', ind )
    decreased = False
    increased = False
    first = True

    while first or (increased or decreased):
        print('Start time CloudSat: ',files_cloudsat[ind+time_displacement].start_time)
        print('Start time ECMWF: ',files_ecmwf[ind+temp_displacement].start_time)
        #print('Time_displacement at start: ',time_displacement)
        #print('Temp_displacement at start: ', temp_displacement)
        delta_start = (files_cloudsat[ind + time_displacement].start_time - files_ecmwf[ind+temp_displacement].start_time).total_seconds()
        #print('Delta start: ',delta_start)

        if delta_start < 0 and not increased:
            time_displacement = time_displacement +1
            decreased = True
            print('Updated time_displacement: ', time_displacement)
        elif delta_start > time_for_one_rotation/2 and not decreased:
            temp_displacement = temp_displacement +1
            increased = True
            print('Updated temp_displacement: ', temp_displacement)
        else:
            increased = False
            decreased = False
        first = False


        #print('CloudSat start time ', files_cloudsat[ind + time_displacement].start_time)
        #print('ECMWF start time', files_ecmwf[ind+temp_displacement].start_time)
        file_ecmwf = files_ecmwf[ind+temp_displacement].open()
        file_cloudsat = files_cloudsat[ind + time_displacement].open()

        temperatures = file_ecmwf.ecmwf_temperature
        latitude_ecmwf = file_ecmwf.latitude
        longitude_ecmwf = file_ecmwf.longitude
        latitude_cs = file_cloudsat.latitude
        longitude_cs  = file_cloudsat.longitude
        altitude_ecmwf = file_ecmwf.EC_height

        number_measurements_cs = len(latitude_cs)
        number_measurements_ecmwf = len(latitude_ecmwf)

        total_time_cs = (file_cloudsat.end_time - file_cloudsat.start_time).total_seconds()
        total_time_ecmwf = (file_ecmwf.end_time - file_ecmwf.start_time).total_seconds()

        measurement_frequency_cs = total_time_cs/number_measurements_cs
        measurement_frequency_ecmwf = total_time_ecmwf/number_measurements_ecmwf

        #print('Measurement frequencies!')
        #print('CloudSat ',measurement_frequency_cs)
        #print('ECMWF ',measurement_frequency_ecmwf)

        '''
        print('Max and min of latitudes for ecmwf and cs')
        print(np.max(latitude_ecmwf))
        print(np.max(latitude_cs))
        print(np.min(latitude_ecmwf))
        print(np.min(latitude_cs))

        print('Max and min of longitudes')
        print(np.max(longitude_ecmwf))
        print(np.max(longitude_cs))
        print(np.min(longitude_ecmwf))
        print(np.min(longitude_cs))


        print(temperatures.shape)
        print(longitude_ecmwf.shape)
        print(latitude_ecmwf.shape)
        print(latitude_cs.shape)
        indexes_zero = np.zeros([len(temperatures),1])
        
        print('Coordinates at start')
        print(latitude_cs[0])
        print(latitude_ecmwf[0])
        print(longitude_cs[0])
        print(longitude_ecmwf[0])

        print('Coordinates after 80 measurements')
        print(latitude_cs[80])
        print(latitude_ecmwf[80])
        print(longitude_cs[80])
        print(longitude_ecmwf[80])
        '''

        min_index = np.argmin(np.absolute(latitude_ecmwf[:] - latitude_cs[0]) + np.absolute(np.multiply(longitude_ecmwf[:],np.cos(np.pi/180*latitude_ecmwf[:])) - longitude_cs[0]*np.cos(np.pi/180*latitude_cs[0])))

        #print('Min_index: ',min_index)

        #print(longitude_ecmwf[min_index])
        #print(longitude_cs[0])
        #print(latitude_ecmwf[min_index])
        #print(latitude_cs[0])

        error_longitude = longitude_ecmwf[min_index] - longitude_cs[0]
        error_latitude = latitude_ecmwf[min_index]-latitude_cs[0]
        #print('Error longitude: ',error_longitude)
        #print('Error latitude: ',error_latitude)
        '''
        for k in range(0,10000,2000):
            difference_longitude = longitude_ecmwf[min_index+k] - longitude_cs[0+k]
            difference_latitude = latitude_ecmwf[min_index+k] - latitude_cs[0+k]
            print('Longitude difference at k=', k,' is:', difference_longitude )
            print('Latitude difference at k=', k, ' is:', difference_latitude)

        difference_longitude = longitude_ecmwf[min_index + len(longitude_cs)-1] - longitude_cs[0 + len(longitude_cs)-1]
        difference_latitude = latitude_ecmwf[min_index + len(longitude_cs)-1] - latitude_cs[0 + len(longitude_cs)-1]
        print('Longitude difference at last measurement is:', difference_longitude)
        print('Latitude difference at last measurement is:', difference_latitude)
        '''
        '''
        #if error_longitude < -0.1 and longitude_ecmwf[min_index] < 0 and longitude_cs[0] > 0 and not decreased:
        #    temp_displacement = temp_displacement+1
        #    print('Updated temp displacement: ', temp_displacement)
        #    decreased = True
        if error_longitude < -0.1 and not decreased:
            time_displacement =time_displacement+1
            print('Updated time displacement: ',time_displacement)
            increased = True
        #elif error_longitude > 0.1 and longitude_ecmwf[min_index] > 0 and longitude_cs[0] < 0 and not increased:
        #    time_displacement =time_displacement+1
        #    print('Updated time displacement: ',time_displacement)
        #    increased = True
        elif error_longitude > 0.1 and not increased:
            temp_displacement = temp_displacement+1
            print('Updated temp displacement: ', temp_displacement)
            decreased = True
        else:
            increased = False
            decreased = False
        first = False
    '''
    displacement_vector_cs = np.append(displacement_vector_cs,ind+time_displacement)
    print(displacement_vector_cs.shape)
    print('Index for CloudSat ',ind+time_displacement)
    print('Index for ECMWF ',ind+temp_displacement)
    displacement_vector_ecmwf = np.append(displacement_vector_ecmwf, ind+temp_displacement)
    min_index_vector = np.append(min_index_vector,min_index)
    #file_ecmwf.close()
    #file_cloudsat.close()
    '''
    temperatures = temperatures[min_index: len(temperatures),:]
    temperatures = temperatures[0:len(temperatures)-len(temperatures)%64,:]
    temperatures = temperatures.reshape(-1,64,125)

    latitude_ecmwf = latitude_ecmwf[min_index:len(latitude_ecmwf)]
    latitude_ecmwf = latitude_ecmwf[0:len(latitude_ecmwf)-len(latitude_ecmwf)%64]
    latitude_ecmwf = latitude_ecmwf.reshape(-1,64)

    longitude_ecmwf = longitude_ecmwf[min_index:len(longitude_ecmwf)]
    longitude_ecmwf = longitude_ecmwf[0:len(longitude_ecmwf)-len(longitude_ecmwf)%64]
    longitude_ecmwf = longitude_ecmwf.reshape(-1,64)

    latitude_cs = latitude_cs[0:len(latitude_cs)-len(latitude_cs)%64]
    longitude_cs = longitude_cs[0:len(longitude_cs)-len(longitude_cs)%64]

    latitude_cs = latitude_cs.reshape(-1,64)
    longitude_cs = longitude_cs.reshape(-1,64)
    print('New shapes!')
    print(temperatures.shape)
    print(latitude_ecmwf.shape)
    print(latitude_cs.shape)
    print(longitude_ecmwf.shape)
    print(longitude_cs.shape)

    print(latitude_ecmwf[0:1])
    print(latitude_cs[0:1])
    print(longitude_ecmwf[0:1])
    print(longitude_cs[0:1])

    print(altitude_ecmwf.shape)
    for i in range(0,1):
        print(altitude_ecmwf)


    for i in range(0,len(temperatures)):
        if i%1000 == 0:

            #print('latitude ecwdadsa', latitude_cs[i] - latitude_ecwf[i+diplacement])
            print('i-1: ', i-1)
            print('latitude', indexes_zero[i-1])



        for j in range(len(temperatures[0])-2,-1,-1):
            if (temperatures[i,j]-273.15)*(temperatures[i,j+1]-273.15)<0:
                #print(i)
                #print(j)
                indexes_zero[i] = j+1
                break
            if j == 0:
                if np.max(temperatures[i])<273.15:
                    indexes_zero[i] = 124
                else:
                    indexes_zero[i]=0
                    print('All above zero: ',temperatures[i])

    #print(indexes_zero[0:64])
    print(np.mean(indexes_zero))
    print(np.std(indexes_zero))
    '''

    if ind%100 == 0 or ind == 4884:
        file_string = 'ecmwf_sync_list_number'+str(ind)+'.h5'

        hf = h5py.File(file_string, 'w')
        hf.create_dataset('displacement_vector_cs', data=displacement_vector_cs)
        hf.create_dataset('displacement_vector_ecmwf', data=displacement_vector_ecmwf)
        hf.create_dataset('min_index_vector', data=min_index_vector)
        hf.create_dataset('time_displacement', data=time_displacement)
        hf.create_dataset('temp_displacement', data=temp_displacement)
        hf.close()
