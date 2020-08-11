import h5py
import torch
import  numpy as np
import matplotlib.pyplot as plt
from os import path
#from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import integrate



def IceWaterPathMethod(cloudsat_scenes):
    scene_number = 0
    Z_m95 = np.power(10, cloudsat_scenes * 0.1)
    # Calculate IWC in g/m^3
    IWC = 0.149 * np.power(Z_m95, 0.681)
    IWC_ver2 = 0.198 * np.power(Z_m95, 0.701)
    IWC_ver3 = 0.108 * np.power(Z_m95, 0.770)
    print('IWC shape ', IWC.shape)

    # print('cloudsat ',example_scene[0])
    # print('IWC ',example_scene_IWC[0])

    # IWC_vertical_integrals = np.multiply(example_scene_IWC,240)
    # IWC_vertical_integrals_ver2 = np.multiply(example_scene_IWC_ver2,240)
    # IWC_vertical_integrals_ver3 = np.multiply(example_scene_IWC_ver3,240)

    IWC_vertical_integrals = np.multiply(IWC, 240)
    IWC_vertical_integrals_ver2 = np.multiply(IWC_ver2, 240)
    IWC_vertical_integrals_ver3 = np.multiply(IWC_ver3, 240)
    # Calculate IWP from IWC
    IWC_vertical_sum = torch.sum(IWC_vertical_integrals, dim=3)  # dim=1 for one scene
    IWC_vertical_sum_ver2 = torch.sum(IWC_vertical_integrals_ver2, dim=3)
    IWC_vertical_sum_ver3 = torch.sum(IWC_vertical_integrals_ver3, dim=3)
    # Convert from
    IWP_test_scene = IWC_vertical_sum * 1e-3
    IWP_test_scene_ver2 = IWC_vertical_sum_ver2 * 1e-3
    IWP_test_scene_ver3 = IWC_vertical_sum_ver3 * 1e-3

    return [IWP_test_scene, IWP_test_scene_ver2, IWP_test_scene_ver3]