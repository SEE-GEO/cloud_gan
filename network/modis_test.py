import os
import shutil
import numpy as np
from os import path
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

index_modis = Index.load(os.path.join(dendrite_path,
                                      "SatData/CloudSat/modis_aux.index"))

index_cloudsat = Index.load(os.path.join(dendrite_path,
                                    "SatData/CloudSat/2b_geoprof.index"))
start = datetime(2015, 1, 1)
end = datetime(2016, 1, 1)

files_modis = index_modis.get_files("CloudSat_Modis_Aux",
                                    start=start,
                                    end=end)

files_cloudsat = index_cloudsat.get_files("CloudSat_2b_GeoProf",
                                         start=start,
                                       end=end)


files_cloudsat.sort(key= lambda x: x.start_time)
files_modis.sort(key= lambda x: x.start_time)


seen_start_times = set()
new_list = []
for obj in files_modis:
    if obj.start_time not in seen_start_times:
        new_list.append(obj)
        seen_start_times.add(obj.start_time)
files_modis=new_list

new_files_cloudsat = []
keep_index = 0
while keep_index <= len(files_cloudsat)-2:
    sec_between_orbits = (files_cloudsat[keep_index + 1].start_time - files_cloudsat[keep_index].start_time).total_seconds()
    if sec_between_orbits < 5000:
        new_files_cloudsat.append(files_cloudsat[keep_index])
        keep_index = keep_index + 1
    else:
        new_files_cloudsat.append(files_cloudsat[keep_index])
    keep_index = keep_index+1

files_cloudsat = new_files_cloudsat


'''
short_counter = 0
long_counter = 0
print('CloudSat dates')
#for i in range(1000, len(files_cloudsat)-1):
for i in range(1000, 1050):
    date_cloudsat = files_cloudsat[i].start_time
    sec_between_orbits = (files_cloudsat[i + 1].start_time - files_cloudsat[i].start_time).total_seconds()
    length_of_file = (files_cloudsat[i+1].end_time - files_cloudsat[i+1].start_time).total_seconds()
    if sec_between_orbits > 6000:
        long_counter = long_counter +1
        print('ind ' + str(i+1) + '-' + str(i) + ': seconds between start of orbits '  +str(sec_between_orbits) + ': too long')
        print('Start time ind', i - 1, files_cloudsat[i - 1].start_time)
        print('Start time ind ', i, date_cloudsat)
        print('Start time ind', i+1, files_cloudsat[i+1].start_time)
        print('Start time ind', i + 2, files_cloudsat[i + 2].start_time)
        print('Start time ind', i + 3, files_cloudsat[i + 3].start_time)
       #print('Length of file ' + str(i+1) + ' in seconds: '+str(length_of_file))
    elif sec_between_orbits < 5000:
        short_counter = short_counter +1
        print('ind ' + str(i + 1) + '-' + str(i) + ': seconds between start of orbits ' + str(sec_between_orbits) + ': too short')
        #print('Length of file ' + str(i+1) + ' in seconds: ' + str(length_of_file))

    #print(str(i) + ': ' + str(date_cloudsat))
    #print(str(i+1) + '-' + str(i) + ': '  +str(sec_between_orbits))
print('Number of wrong (short) files ' +str(short_counter))
print('Number of wrong (long) files ' +str(long_counter))

print(len(files_cloudsat))
print(len(files_modis))
'''

if path.exists('rr_modis_cloudsat_data_2015_1416bla.h5'):
    hf = h5py.File('./rr_modis_cloudsat_data_2015_1416.h5','r')
    displacement = (np.array(hf.get('displacement')))
else:
    displacement = 0

time_for_one_rotation = (files_cloudsat[1].start_time - files_cloudsat[0].start_time).total_seconds()
#for ind in range(0, len(files_cloudsat)):
for ind in range(0, 10):
    print(' -- NEW LAP! Index', ind )
    decreased = False
    increased = False
    first = True
    while first or (increased or decreased):
        delta_start = (files_cloudsat[ind].start_time - files_modis[ind + displacement].start_time).total_seconds()
        if delta_start < 0 and not increased:
            displacement = displacement - 1
            decreased = True
        elif delta_start > time_for_one_rotation and not decreased:

            displacement = displacement + 1
            increased= True
        else:
            increased = False
            decreased = False
        first = False
    print('CloudSat start time ', files_cloudsat[ind].start_time)
    print('Modis start time', files_modis[ind + displacement].start_time)
    file_modis = files_modis[ind + displacement].open()
    file_cs = files_cloudsat[ind].open()

    latitudes = (file_modis.modis_latitude)
    longitudes = (file_modis.modis_longitude)
    cs_latitudes = file_cs.latitude
    cs_longitudes = file_cs.longitude

    dt = (file_modis.end_time - file_modis.start_time).total_seconds() / len(latitudes)
    print('dt ', dt)
    delta_start = (file_cs.start_time - file_modis.start_time).total_seconds()
    print('delta start ', delta_start)
    n_exact = ((delta_start / dt))
    time_for_modis = (file_modis.end_time - file_modis.start_time).total_seconds()/len(file_modis.modis_latitude)
    time_for_cs = (file_cs.end_time - file_cs.start_time).total_seconds()/len(file_cs.latitude)
    delta_freq =  (time_for_modis - time_for_cs)/time_for_modis
    print('deltafreq ', delta_freq)
    n_exact = n_exact - delta_freq*n_exact #add +5 after index 1001
    print('n vid start', n_exact)
    n= int(round(n_exact))
    if n < 37082:
        min_index = np.argmin(np.absolute(latitudes[n] - cs_latitudes[0]) + np.absolute(np.multiply(longitudes[n],np.cos(np.pi/180*latitudes[n])) - cs_longitudes[0]*np.cos(np.pi/180*cs_latitudes[0])))
        print('difference in longitude: ', ((longitudes[n][min_index]*np.cos(np.pi/180*latitudes[n][min_index]) - cs_longitudes[0]*np.cos(np.pi/180*cs_latitudes[0]))* 111) , ' km')
        print('difference in latitude: ', (latitudes[n][min_index] - cs_latitudes[0])*111  , ' km')

        n_start = n
        n_low = n_start
        n_high = n_start
        min_index_low = np.argmin(np.absolute(latitudes[n-1] - cs_latitudes[0]) + np.absolute(np.multiply(longitudes[n-1],np.cos(np.pi/180*latitudes[n-1])) - cs_longitudes[0]*np.cos(np.pi/180*cs_latitudes[0])))
        min_index_high = np.argmin(np.absolute(latitudes[n+1] - cs_latitudes[0]) + np.absolute(np.multiply(longitudes[n+1],np.cos(np.pi/180*latitudes[n+1])) - cs_longitudes[0]*np.cos(np.pi/180*cs_latitudes[0])))
        delta_latitude = abs((latitudes[n_start][min_index]-cs_latitudes[0])*111)
        delta_latitude2 = abs((latitudes[n_start-1][min_index_low]-cs_latitudes[0])*111)
        delta_longitude = abs((longitudes[n_start][min_index]*np.cos(np.pi/180*latitudes[n_start][min_index]) - cs_longitudes[0]*np.cos(np.pi/180*cs_latitudes[0]))* 111)
        delta_longitude2 = abs((longitudes[n_start-1][min_index_low] * np.cos(np.pi / 180 * latitudes[n_start-1][min_index_low]) -cs_longitudes[0] * np.cos(np.pi / 180 * cs_latitudes[0])) * 111)
        while delta_latitude2 + delta_longitude2< delta_latitude+delta_longitude:
            print('Loop low')
            n_low = n_low-1
            min_index_low = np.argmin(np.absolute(latitudes[n_low] - cs_latitudes[0]) + np.absolute(np.multiply(longitudes[n_low], np.cos(np.pi / 180 * latitudes[n_low])) - cs_longitudes[0] * np.cos(np.pi / 180 * cs_latitudes[0])))
            delta_latitude=delta_latitude2
            delta_latitude2 =abs((latitudes[n_low-1][min_index_low]-cs_latitudes[0])*111)
            delta_longitude = delta_longitude2
            delta_longitude2 = abs((longitudes[n_low - 1][min_index_low] * np.cos(np.pi / 180 * latitudes[n_low - 1][min_index_low]) - cs_longitudes[0] * np.cos(np.pi / 180 * cs_latitudes[0])) * 111)

        delta_latitude = abs((latitudes[n_start][min_index] - cs_latitudes[0])*111)
        delta_latitude2 = abs((latitudes[n_start + 1][min_index_high] - cs_latitudes[0])*111)
        delta_longitude = abs((longitudes[n_start][min_index] * np.cos(np.pi / 180 * latitudes[n_start][min_index]) - cs_longitudes[0] * np.cos(np.pi / 180 * cs_latitudes[0])) * 111)
        delta_longitude2 = abs((longitudes[n_start + 1][min_index_high] * np.cos(np.pi / 180 * latitudes[n_start + 1][min_index_high]) - cs_longitudes[0] * np.cos(np.pi / 180 * cs_latitudes[0])) * 111)
        while delta_latitude2 + delta_longitude2 < delta_latitude + delta_longitude:
            print('Loop high')
            n_high = n_high +1
            min_index_high = np.argmin(np.absolute(latitudes[n_high] - cs_latitudes[0]) + np.absolute(np.multiply(longitudes[n_high], np.cos(np.pi / 180 * latitudes[n_high])) - cs_longitudes[0] * np.cos(np.pi / 180 * cs_latitudes[0])))
            delta_latitude = delta_latitude2
            delta_latitude2 = abs((latitudes[n_high + 1][min_index_high] - cs_latitudes[0]) * 111)
            delta_longitude = delta_longitude2
            delta_longitude2 = abs((longitudes[n_high + 1][min_index_high] * np.cos(np.pi / 180 * latitudes[n_high + 1][min_index_high]) - cs_longitudes[0] * np.cos(np.pi / 180 * cs_latitudes[0])) * 111)

        if abs((latitudes[n_low][min_index_low]-cs_latitudes[0])*111) + abs((longitudes[n_low][min_index_low]*np.cos(np.pi/180*latitudes[n_low][min_index_low]) - cs_longitudes[0]*np.cos(np.pi/180*cs_latitudes[0]))* 111) < \
                abs((latitudes[n_start][min_index]-cs_latitudes[0])*111) + abs((longitudes[n_start][min_index]*np.cos(np.pi/180*latitudes[n_start][min_index]) - cs_longitudes[0]*np.cos(np.pi/180*cs_latitudes[0]))* 111):
            n = n_low
            min_index = min_index_low
        elif abs((latitudes[n_high][min_index_high]-cs_latitudes[0])*111) + abs((longitudes[n_high][min_index_high]*np.cos(np.pi/180*latitudes[n_high][min_index_high]) - cs_longitudes[0]*np.cos(np.pi/180*cs_latitudes[0]))* 111) < \
                abs((latitudes[n_start][min_index]-cs_latitudes[0])*111) + abs((longitudes[n_start][min_index]*np.cos(np.pi/180*latitudes[n_start][min_index]) - cs_longitudes[0]*np.cos(np.pi/180*cs_latitudes[0]))* 111):
            n = n_high
            min_index = min_index_high

        print('Uppdated difference in longitude: ', ((longitudes[n][min_index] * np.cos(np.pi / 180 * latitudes[n][min_index]) - cs_longitudes[0] * np.cos(np.pi / 180 * cs_latitudes[0])) * 111), ' km')
        print('Uppdated difference in latitude: ', (latitudes[n][min_index] - cs_latitudes[0]) * 111, ' km')
        if ind%1 == 0:
            print('index ' + str(ind))



        # f,ax = plt.subplots(1, 1)
        # pcm = ax.pcolormesh(np.reshape(radiances[0][0],(3,5)))
        # cb=f.colorbar(pcm,ax=ax)
        # plt.show()
        # files[ind].close()



        y = file_cs.longitude
        z = file_cs.altitude
        x = file_cs.latitude
        x = np.broadcast_to(x.reshape(-1, 1), z.shape)
        y = np.broadcast_to(y.reshape(-1,1),z.shape)
        position_x = x[0:len(z) - len(z) % 64, 0].copy()
        position_y = y[0:len(z) - len(z) % 64, 0].copy()
        print('initial latitude shape',position_x.shape)
        loop_start = file_cs.start_time
        loop_end = file_cs.end_time
        rr = file_cs.radar_reflectivity
        DEM_elevation = file_cs.DEM_elevation
        cloud_mask = file_cs.cloud_mask
        emissivity_modis = (file_modis.emissivity_channels)[1:11]

        print('emissivity_modis', emissivity_modis.shape)
        emissivity = np.ones([(len(z)//64), 64, 10])*(-40)
        emissivity_mask = emissivity >= 32768
        emissivity = ma.masked_array(emissivity,mask=emissivity_mask)
        rr_data_element = np.ones([(len(z) // 64), 64, 64]) * (-40)
        mask = rr_data_element == -8888
        rr_data_element = ma.masked_array(rr_data_element, mask=mask, dtype=np.float32)
        bad_positions = []
        # should we change the mask?
        for i in range(0, len(z) - 64, 64):
            n_cloud_free_tiles = 0
            elevation_too_high = False
            j = np.ones(64, dtype=np.int8) * 124
            test = (int)(i / 64)
            for k in range(0, 64):

                if DEM_elevation[i + k] > 500:
                    elevation_too_high = True
                    break
                else:

                    while z[i + k, j[k]] < 1000:
                        j[k] = j[k] - 1

                    n_cloud_free_tiles = n_cloud_free_tiles + np.sum(cloud_mask[i + k][j[k]:j[k] - 63:-1] < 20)

            if n_cloud_free_tiles > 64 * 64 * 0.7 or elevation_too_high:
                position_x[i:i + 64] = 4444
                position_y[i:i + 64] = 4444
                rr_data_element[test][:][:] = 4444
                emissivity[test][:][:] = 4444
            else:
                missmatch_position =0
                for ii in range(0, 64):
                    rr_data_element[test][ii] = rr[i + ii][j[ii]:j[ii] - 64:-1]

                    min_index_position = np.argmin(np.absolute(latitudes[n + i +ii] - cs_latitudes[i+ii]) +
                                                   np.absolute(np.multiply(longitudes[n + i+ii],np.cos(np.pi/180*latitudes[n+i+ii])) - cs_longitudes[i+ii]*np.cos(np.pi/180*cs_latitudes[i+ii])))
                    #print('difference in longitude: ', (np.multiply(longitudes[n + i+ii],np.cos(np.pi/180*latitudes[n+i+ii])) - cs_longitudes[i+ii]*np.cos(np.pi/180*cs_latitudes[i+ii]))* 111, ' km')
                    #print('difference in latitude: ', (latitudes[n+i+ii][min_index_position] - cs_latitudes[i+ii])*111  , ' km')
                    if abs((longitudes[n + i+ii][min_index_position]*np.cos(np.pi/180*latitudes[n+i+ii][min_index_position]) - cs_longitudes[i+ii]*np.cos(np.pi/180*cs_latitudes[i+ii]))* 111) >0.7 or abs((latitudes[n+i+ii][min_index_position] - cs_latitudes[i+ii])*111) >0.7:
                       for channel in range(0,10):
                            emissivity[test][ii][channel]=emissivity_modis[channel][i + ii + n][min_index_position]
                            #print('when wrong match in coordinate ','a,b,i ',test, ' ', ii, ' ', channel, ' ',emissivity[test][ii][channel])
                       bad_positions.append([test,ii])

                       missmatch_position +=1

                        #print('longitude',abs((longitudes[n+i+ii][min_index_position] - cs_longitudes[i+ii])*np.cos(cs_latitudes[i+ii])* 111))
                        #print('latitude',abs((latitudes[n+i+ii][min_index_position] - cs_latitudes[i+ii])*111))


                    else:
                        for channel in range(0,10):
                            emissivity[test][ii][channel]=emissivity_modis[channel][i + ii + n][min_index_position]

                    if missmatch_position > 21:
                        print('scene removed by wrong position',test)
                        position_x[i:i + 64] = 4444
                        position_y[i:i + 64] = 4444
                        rr_data_element[test][:][:] = 4444
                        emissivity[test][:][:] = 4444
                        break

        rr_data_element = rr_data_element[rr_data_element != 4444].reshape(-1, 64, 64)
        position_x = position_x[position_x != 4444].reshape(-1, 1)
        position_y = position_y[position_y != 4444].reshape(-1, 1)
        emissivity = emissivity[emissivity != 4444].reshape(-1,64,10)

        #print('Longitude for scene (CloudSat): '+str(cs_longitudes[i]))
        #print('Latitude for scene (CloudSat): '+str(cs_latitudes[i]))

        if (True):
            rr_data_element[rr_data_element > 20] = 20
            rr_data_element[rr_data_element < -35] = -35
            rr_data_element = 2 * (rr_data_element + 35) / 55 - 1
        print(' - emissivity final shape', emissivity.shape)
        start_date_string = loop_start.strftime("%Y-%m-%d %H:%M:%S")
        end_date_string = loop_end.strftime("%Y-%m-%d %H:%M:%S")
        print('latitude shape ',position_x)
        print('longitude shape ', position_y.shape)
        name_string = start.strftime("%Y")
        filename = 'rr_modis_cloudsat_data_testesttest' + name_string + '_' + str(ind).zfill(4) + '.h5'
        hf = h5py.File(filename, 'w')
        hf.create_dataset('rr', data=rr_data_element)
        hf.create_dataset('start', data=start_date_string)
        hf.create_dataset('end', data=end_date_string)
        hf.create_dataset('latitude', data=position_x)
        hf.create_dataset('longitude', data=position_y)
        hf.create_dataset('emissivity', data=emissivity)
        hf.create_dataset('displacement', data=displacement)
        hf.create_dataset('bad_positions', data=bad_positions)
        hf.close()
        print(n_exact)
        # print(len(rr_data_element))
    else:
        print('missing data point')










