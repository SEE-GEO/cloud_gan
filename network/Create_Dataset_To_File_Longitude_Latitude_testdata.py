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


def create_dataset_to_file_longitude_latitude_testdata(rescale):
    dendrite_path = "~/Dendrite"
    index = Index.load(os.path.join(dendrite_path,
                            "SatData/CloudSat/2b_geoprof.index"))

    start = datetime(2015, 1, 1)
    end = datetime(2016, 1, 1)

    files = index.get_files("CloudSat_2b_GeoProf",
                            start=start,
                            end=end)

    hf = h5py.File('./rr_data/index_list.h5', 'r')
    index_list = (np.array(hf.get('index_list')))
    for ind in index_list:
        file = files[ind].open()

        z = file.altitude
        x = file.latitude
        y = file.longitude
        x = np.broadcast_to(x.reshape(-1, 1), z.shape)
        y = np.broadcast_to(y.reshape(-1, 1), z.shape)
        position = x[0:len(z) - len(z) % 64, 0].copy()
        position_y = y[0:len(z) - len(z) % 64, 0].copy()
        loop_start = file.start_time
        loop_end = file.end_time
        rr = file.radar_reflectivity
        DEM_elevation = file.DEM_elevation
        DEM_elevation = DEM_elevation[0:len(z)-len(z)%64,0]
        cloud_mask = file.cloud_mask

        rr_data_element = np.ones([(len(z) // 64), 64, 64]) * (-40)
        mask = rr_data_element == -8888
        rr_data_element = ma.masked_array(rr_data_element, mask=mask, dtype=np.float32)
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
                position[i:i + 64] = 4444
                position_y[i:i + 64] = 4444
                rr_data_element[test][:][:] = 4444
                DEM_elevation[i:i+64] = 4444
            else:
                for ii in range(0, 64):
                    rr_data_element[test][ii] = rr[i + ii][j[ii]:j[ii] - 64:-1]


        rr_data_element = rr_data_element[rr_data_element != 4444].reshape(-1, 64, 64)
        position = position[position != 4444].reshape(-1, 1)
        position_y = position_y[position_y != 4444].reshape(-1, 1)
        DEM_elevation = DEM_elevation[DEM_elevation != 4444].reshape(-1, 1)
        if (rescale):
            rr_data_element[rr_data_element > 20] = 20
            rr_data_element[rr_data_element < -35] = -35
            rr_data_element = 2 * (rr_data_element + 35) / 55 - 1

        start_date_string = loop_start.strftime("%Y-%m-%d %H:%M:%S")
        end_date_string = loop_end.strftime("%Y-%m-%d %H:%M:%S")

        name_string = start.strftime("%Y")
        filename = 'rr_data_latitude_longitude_elevation' + name_string + '_' + str(ind).zfill(4) + '.h5'
        hf = h5py.File(filename, 'w')
        hf.create_dataset('rr', data=rr_data_element)
        hf.create_dataset('start', data=start_date_string)
        hf.create_dataset('end', data=end_date_string)
        hf.create_dataset('latitude', data=position)
        hf.create_dataset('longitude', data=position_y)
        hf.create_dataset('DEM_elevation', data=DEM_elevation)
        hf.close()
        # print(len(rr_data_element))
        if (ind % 1 == 0):
            print(ind)
