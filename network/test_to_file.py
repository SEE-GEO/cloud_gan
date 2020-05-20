import os
import shutil
import numpy as np

import numpy.ma as ma
from datetime import datetime
import pandas as pd
import h5py
import torch

#from wxdata import Index

from Create_Dataset_To_File import create_dataset_to_file

create_dataset_to_file(True)

#hf = h5py.File('rr_data_2015_0140.h5', 'r')
#rr=hf.get('rr')
#file_string = 'rr_data_2015_' + str(0).zfill(4) +'.h5'
#hf = h5py.File(file_string, 'r')


#dendrite_path = "~/Dendrite"
#index = Index.load(os.path.join(dendrite_path, "SatData/CloudSat/2b_geoprof.index"))

#start = datetime(2015, 1, 1)
#end   = datetime(2016, 1, 1)

#files = index.get_files("CloudSat_2b_GeoProf",start = start,end = end)
#file = files[0].open()
#rr = file.radar_reflectivity
#rr = np.array(rr)
# start = hf.get('start')
# end = hf.get('end')
# print(np.array(start))
# print(np.array(end))








# files[ind].close()
