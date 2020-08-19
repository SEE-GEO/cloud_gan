import numpy as np
import numpy.ma as ma
import matplotlib
import matplotlib.pyplot as plt
from scipy import constants
import h5py
import pysal
from esda.moran import Moran
import torch
from IceWaterPathMethod import IceWaterPathMethod

hft = h5py.File('./rr_data/cloudsat_training_data_conc.h5', 'r')
hf2 = h5py.File('./IAAFT_generated_conc.h5', 'r')
iaaft_scenes = np.array(hf2.get('iaaft_scenes'))
print('IAAFT scenes ',iaaft_scenes.shape)
template = hft.get('cloudsat_scenes')
template = np.array(template)
template = template.reshape(-1,64,64)
print('Template shape' ,template.shape)
iaaft_scenes = np.float32(iaaft_scenes)

iaaft_scenes = (iaaft_scenes+1)*(55/2)-35
template = (template+1)*(55/2)-35

w = pysal.lib.weights.lat2W(64, 64, rook=True)  # use rook=False to add corners to neighbours

#Plot example scenes in section below
xplot=np.linspace(0,64*1.1,64)
yplot=np.linspace(1,16.36,64)
from matplotlib.colors import Normalize

f,axs = plt.subplots(4,5, figsize=(13,9))
#ax=axs[range(0,3),range(0,3)]
norm = Normalize(-35, 20)
step = 8 #step = 8 for fall streaks
for i in range(0,5):
    Z = template[i*step]
    X = iaaft_scenes[i*step]
    mi = Moran(Z, w)
    mix = Moran(X,w)
    value = "%.3f" % mi.I
    valuex ="%.3f" % mix.I

    pcm1 = axs[3,i].pcolormesh(xplot,yplot,np.transpose(Z), norm=norm)
    axs[3, i].minorticks_on()
    axs[3,i].set_xlabel('Position [km]', fontsize=12)
    if i == 0:
        axs[3, i].set_ylabel('Real \nAltitude [km]', fontsize=12)
    else:
        axs[3, i].tick_params(labelleft=False)
    axs[3, i].tick_params(axis='both', which='both', labelsize='12')
    axs[3, i].text(2, 15, 'I = '+str(value), style='italic', fontsize=12, color='white')

    pcm = axs[2,i].pcolormesh(xplot,yplot,np.transpose(X), norm=norm)
    axs[2, i].minorticks_on()
    axs[2, i].tick_params(axis='both', which='major', labelsize='12')
    axs[2, i].tick_params(axis='both', which='minor')
    axs[2, i].tick_params(labelbottom=False)
    axs[2, i].text(2, 15, 'I = '+str(valuex), style='italic', fontsize=12, color='white')
    if i == 0:
        axs[2, i].set_ylabel('Generated \nAltitude [km]', fontsize=12)
    else:
        axs[2, i].tick_params(labelleft=False)


    [IWP_cs_1, IWP_cs_2, IWP_cs_3] = IceWaterPathMethod(torch.tensor(template[i*step]))
    [IWP_generated_1, IWP_generated_2, IWP_generated_3] = IceWaterPathMethod(torch.tensor(iaaft_scenes[i*step]))
    axs[0, i].minorticks_on()
    axs[0, i].tick_params(axis='both', which='major', labelsize='12')
    axs[0, i].tick_params(axis='both', which='minor')
    axs[0,i].set_ylim([0.0625,16])
    pcm = axs[0, i].semilogy(xplot, (IWP_generated_1), color='grey', label='IWP generated',basey=2)
    pcm = axs[0, i].semilogy(xplot, (IWP_cs_1), color='black', label='IWP real',basey=2)
    title_str = 'IWP generated' + str(i)
    #axs[0,i].set_title(title_str, fontsize=2)
    axs[0, i].tick_params(labelbottom=False)
    if i == 0:
        axs[0, i].set_ylabel('IWP [g m⁻²]', fontsize=12)
    else:
        axs[0, i].tick_params(labelleft=False)

    # Calculate cloud top height:
    altitudes_template = np.zeros([1, 64])
    altitudes_iaaft = np.zeros([1, 64])
    bottom_template = np.zeros([1, 64])
    bottom_iaaft = np.zeros([1, 64])
    for k in range(0, 64):
        first_cloud_location_cs = 0
        first_cloud_location_iaaft = 0
        for j in range(63, -1, -1):  # Loop backwards, starting from top of scene
            if template[i*step, k, j] >= -20 and first_cloud_location_cs == 0:  # Set dBZ limit for cloud top detection
                altitudes_template[0, k] = j  # save the index of the altitude where the cloud top is for each position
                first_cloud_location_cs = +1
            if iaaft_scenes[i*step, k, j] >= -20 and first_cloud_location_iaaft == 0:
                altitudes_iaaft[0, k] = j
                first_cloud_location_iaaft = +1
    altitudes_template = (altitudes_template * 0.24) + 1  # Altitude of cloud top over sea level, [km]
    print('Altitudes of cloud tops (CloudSat):', altitudes_template, ' [km]')
    altitudes_iaaft = (altitudes_iaaft * 0.24) + 1  # Altitude of cloud top over sea level, [km]
    print('Altitudes of cloud tops (IAAFT):', altitudes_iaaft, ' [km]')

    height = axs[1, i].plot(xplot, np.transpose(altitudes_template), color='black')
    height2 = axs[1, i].plot(xplot, np.transpose(altitudes_iaaft), color='grey')
    height = axs[3, i].plot(xplot, np.transpose(altitudes_template), color='white', linewidth=0.7)
    height2 = axs[2, i].plot(xplot, np.transpose(altitudes_iaaft), color='white', linewidth=0.7)
    axs[1,i].set_ylim([1,16.36])
    axs[1,i].minorticks_on()
    axs[1, i].tick_params(axis='both', which='major', labelsize='12')
    axs[1, i].tick_params(axis='both', which='minor')
    axs[1,i].tick_params(labelbottom=False)
    if i == 0:
        axs[1, i].set_ylabel('Cloud-top \nheight [km]', fontsize=12)
    else:
        axs[1, i].tick_params(labelleft=False)

f.tight_layout()
f.subplots_adjust(right=0.88)
cbar_ax = f.add_axes([0.9, 0.0698, 0.025, 0.443]) #([distance from left, bottom height, width, height])
cbar1= f.colorbar(pcm1, cax=cbar_ax)
cbar1.set_label('Reflectivities [dBZ]', fontsize=12)
cbar1.ax.tick_params(labelsize=12)
plt.savefig('./Results/IAAFT/example_scenes_iaaft2.png')
#plt.show()