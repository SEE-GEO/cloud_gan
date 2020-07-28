import h5py
import torch
import  numpy as np
import matplotlib.pyplot as plt
from os import path
#from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import integrate


print('starting to read concatenated files')
location = './'
file_string = location + 'GAN_test_data_with_elevation_conc' + '.h5'
hf = h5py.File(file_string, 'r')
cloudsat_scenes = torch.tensor(hf.get('cloudsat_scenes_conc'))
print('finished reading cloudsat file')
#DEM_elevation = torch.tensor(hf.get('DEM_elevation_conc'))
#print('finished reading elevation file')

#Choose scene to evaluate
#scene_number = 40 #40, 260, 240, 170, 160
scene_number = [40, 260, 240, 170, 160, 4032, 12600]

example_scene = cloudsat_scenes[scene_number]
#example_scene[example_scene < 0] = 0
print('Shape of cloudsat_scenes ',cloudsat_scenes.shape)
#Transform radar reflectivity from dBZ to linear units (mm^6/m^3)
Z_m95 = np.power(10,cloudsat_scenes*0.1)
#Calculate IWC in g/m^3
IWC = 0.149*np.power(Z_m95,0.681)
IWC_ver2 = 0.198*np.power(Z_m95,0.701)
IWC_ver3 = 0.108*np.power(Z_m95,0.770)
print('IWC shape ',IWC.shape)
example_scene_IWC = IWC[scene_number]
example_scene_IWC_ver2 = IWC_ver2[scene_number]
example_scene_IWC_ver3 = IWC_ver3[scene_number]
#print('cloudsat ',example_scene[0])
#print('IWC ',example_scene_IWC[0])

#IWC_vertical_integrals = np.multiply(example_scene_IWC,240)
#IWC_vertical_integrals_ver2 = np.multiply(example_scene_IWC_ver2,240)
#IWC_vertical_integrals_ver3 = np.multiply(example_scene_IWC_ver3,240)

IWC_vertical_integrals = np.multiply(IWC,240)
IWC_vertical_integrals_ver2 = np.multiply(IWC_ver2,240)
IWC_vertical_integrals_ver3 = np.multiply(IWC_ver3,240)
#Calculate IWP from IWC
IWC_vertical_sum = torch.sum(IWC_vertical_integrals, dim=2) #dim=1 for one scene
IWC_vertical_sum_ver2 = torch.sum(IWC_vertical_integrals_ver2, dim=2)
IWC_vertical_sum_ver3 = torch.sum(IWC_vertical_integrals_ver3, dim=2)
#Convert from
IWP_test_scene = IWC_vertical_sum *1e-3
IWP_test_scene_ver2 = IWC_vertical_sum_ver2*1e-3
IWP_test_scene_ver3 = IWC_vertical_sum_ver3*1e-3
print(IWP_test_scene)
#Check units and multiplication with delta_h for IWP




xplot = np.linspace(0, 64 * 1.1, 64)
yplot = np.linspace(1, 16.36, 64)
figure_string = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6']
axis_string = ['ax0', 'ax1', 'ax2', 'ax3', 'ax4', 'ax5', 'ax6']
from matplotlib.colors import Normalize
for i in range(0,len(scene_number)):
    #f,ax = plt.subplots(3, 1, gridspec_kw=dict(nrows=3, ncols=1, hspace=0.1))
    figure_string[i],axis_string[i] = plt.subplots(3, 1)
    norm = Normalize(-35, 20)
    pcm = (axis_string[i])[0].pcolormesh(xplot,yplot, np.transpose(cloudsat_scenes[scene_number[i]]), norm=norm)
    (axis_string[i])[0].set_xlabel("Position [km]")
    (axis_string[i])[0].set_ylabel("Altitude [km]")
    (axis_string[i])[0].set_title('Scene number ' + str(scene_number[i]))
    cb=figure_string[i].colorbar(pcm,ax=(axis_string[i])[0])
    cb.set_label('Reflectivities [dBZ]')

    #norm2 = Normalize(0,0.149*20**0.681)
    pcm2 = (axis_string[i])[1].pcolormesh(xplot,yplot,np.transpose(IWC[scene_number[i]]))
    (axis_string[i])[1].set_xlabel("Position [km]")
    (axis_string[i])[1].set_ylabel("Altitude [km]")
    cb2=figure_string[i].colorbar(pcm2,ax=(axis_string[i])[1])
    cb2.set_label('IWC ver1 [g m^-3]')

    (axis_string[i])[2].plot(xplot, IWP_test_scene[scene_number[i]], label = 'IWP ver1')
    (axis_string[i])[2].plot(xplot, IWP_test_scene_ver2[scene_number[i]], label = 'IWP ver2')
    (axis_string[i])[2].plot(xplot, IWP_test_scene_ver3[scene_number[i]], label = 'IWP ver3')
    (axis_string[i])[2].set_xlabel("Position [km]")
    (axis_string[i])[2].set_ylabel("IWP [g m^-2]")
    (axis_string[i])[2].legend()

    figure_string[i].tight_layout()
plt.show()