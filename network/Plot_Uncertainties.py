import h5py
from IceWaterPathMethod import IceWaterPathMethod
from GAN_generator import GAN_generator
from plot_cloud import plot_cloud
from os import path
import torch
import numpy as np
import matplotlib.pyplot as plt
from plot_examples_CGAN_method import plot_examples_CGAN_method
from Create_Histogram_CGAN_Method import Create_Histogram_CGAN_Method

hf = h5py.File('Uncertainties_modis', 'r')
max_uncertainty = np.array(hf.get('max_uncertainty'))
min_uncertainty = np.array(hf.get('min_uncertainty'))
average_uncertainty = np.array(hf.get('average_uncertainty'))

print('Maximum uncertainty: ',max_uncertainty)
print('Minimum uncertainty: ',min_uncertainty)
print('Average uncertainty: ',average_uncertainty)

#plot_examples_CGAN_method(max_uncertainty,3)
#plot_examples_CGAN_method(min_uncertainty,1)
#plot_examples_CGAN_method(average_uncertainty,2)

#plot_examples_CGAN_method(max_uncertainty,6)
#plot_examples_CGAN_method(min_uncertainty,4)
#plot_examples_CGAN_method(average_uncertainty,5)

plot_examples_CGAN_method(max_uncertainty,9)
#plot_examples_CGAN_method(min_uncertainty,7)
#plot_examples_CGAN_method(average_uncertainty,8)

#Create_Histogram_CGAN_Method(max_uncertainty,3)
#Create_Histogram_CGAN_Method(min_uncertainty,1)
#Create_Histogram_CGAN_Method(average_uncertainty,2)

plot_examples_CGAN_method(max_uncertainty,12)
plot_examples_CGAN_method(max_uncertainty,15)
plot_examples_CGAN_method(max_uncertainty,18)
'''
for band in range(0, 8):
    modis_uncertainty_per_band = max_uncertainty[band] / 100
    modis_uncertainty_per_band = float(modis_uncertainty_per_band)
    uncertainty = torch.randn(size=[1, 1, 10])
    print(modis_uncertainty_per_band)
    print(uncertainty)
'''