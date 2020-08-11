
import h5py
import torch
import  numpy as np
import matplotlib.pyplot as plt
from os import path
#from mpl_toolkits.axes_grid1 import make_axes_locatable

#Create one file with all cloudsat_scenes
from GAN_generator import GAN_generator

location = './GAN/'
counter = 0
hf = h5py.File('./rr_data/index_list.h5', 'r')
index_list = (np.array(hf.get('index_list')))
for ind in index_list:
    location = './GAN/'
    file_string = location + 'rr_data_latitude_longitude_elevation2015_' + str(ind).zfill(4) +'.h5'
    hf = h5py.File(file_string, 'r')
    cloudsat_scenes_temp = torch.tensor(hf.get('rr')).view(-1,64,64)
    cloudsat_scenes_temp = (cloudsat_scenes_temp + 1)*(55 / 2)-35
    DEM_elevation_temp = torch.tensor(hf.get('DEM_elevation'))
    counter +=1
    print(counter)
    if counter == 1:
        cloudsat_scenes=cloudsat_scenes_temp
        DEM_elevation = DEM_elevation_temp
    else:
        cloudsat_scenes = torch.cat([cloudsat_scenes,cloudsat_scenes_temp],0)
        DEM_elevation = torch.cat([DEM_elevation,DEM_elevation_temp],0)
       # cloudsat_scenes = cloudsat_scenes + cloudsat_scenes_temp
DEM_elevation = DEM_elevation.view(-1,64)
#cloudsat_scenes = cloudsat_scenes.view(-1,64)
print(DEM_elevation.shape)
print(cloudsat_scenes.shape)
ocean_scene_counter = 0
land_scene_counter = 0
ocean_cloudsat = []
land_cloudsat = []
for i in range(0,len(DEM_elevation)):
    ocean_tile_counter = 0
    for j in range(0,64):
        if DEM_elevation[i,j] ==-9999: #if position over ocean
            ocean_tile_counter =ocean_tile_counter+1
    #print(ocean_tile_counter, 'ocean tile counter after inner loop')
    if ocean_tile_counter > 43: #if more than 2/3 of positions in one scene is over ocean tiles
        #print('scene number ', i, ' over ocean')
        ocean_scene_counter = ocean_scene_counter + 1
        if ocean_scene_counter == 1:
            ocean_cloudsat = cloudsat_scenes[i,:,:]
        else:
            ocean_cloudsat = np.append(ocean_cloudsat,cloudsat_scenes[i,:,:],0)
    else:
        print('scene number ', i, ' over land')
        land_scene_counter = land_scene_counter +1
        if land_scene_counter == 1:
            land_cloudsat = cloudsat_scenes[i,:,:]
        else:
            land_cloudsat = np.append(land_cloudsat,cloudsat_scenes[i,:,:],0)
print('Number of scenes over ocean', ocean_scene_counter)
print('Number of scenes over land', land_scene_counter)
print(ocean_cloudsat.shape)
print(land_cloudsat.shape)

num_bins = 200
num_removed = int((30/165)*num_bins)
total_ocean_histogram = np.ones([64,num_bins])
total_land_histogram = np.ones([64,num_bins])

#Ocean and land histograms below
for i in range(0,64):
    ocean_scenes_histogram, bin_edges_ocean = np.histogram(ocean_cloudsat[:, i], bins=num_bins, range=(-34, 20), density=True)
    land_scenes_histogram, bin_edges_land = np.histogram(land_cloudsat[:, i], bins=num_bins, range=(-34, 20), density=True)
    total_ocean_histogram[i] = ocean_scenes_histogram
    total_land_histogram[i] = land_scenes_histogram
print('Ocean histogram shape ',total_ocean_histogram.shape)
print('Land histogram shape ',total_land_histogram.shape)

new_total_ocean = total_ocean_histogram[:,num_removed:num_bins]
print(new_total_ocean.shape)
new_total_land = total_land_histogram[:,num_removed:num_bins]

x_ocean=np.linspace(bin_edges_ocean[num_removed],20,num_bins-num_removed)
y_ocean=np.linspace(1,16.36,64)
#y=np.linspace(1,13,50) #240*50+1000
f1,ax1 = plt.subplots(1, 1, figsize=(10.5,15))
pcm1 = ax1.pcolormesh(x_ocean, y_ocean, new_total_ocean)
ax1.set_ylabel("Altitude [km]", fontsize=28)
ax1.set_xlabel("Reflectivity [dBZ]", fontsize=28)
#ax.set_aspect((num_bins-num_removed_bins)/64)
ax1.set_aspect(135/64)
#ax1.set_title('Ocean', fontsize=14)
cb1=f1.colorbar(pcm1,ax=ax1, orientation = 'horizontal', fraction =0.049, pad=0.15)
cb1.set_label('Norm. occurrence (ocean)', fontsize=28)
ax1.tick_params(axis='both', which='major', labelsize=26)
cb1.set_ticks(np.arange(np.min(new_total_ocean), np.max(new_total_ocean), step=0.01))
cb1.ax.tick_params(labelsize=26, rotation=45)
#cb1.ax.tick_params(labelsize=12)
#f1.tight_layout()
plt.savefig('Ocean_histogram_GAN_new')

x_land=np.linspace(bin_edges_land[num_removed],20,num_bins-num_removed)
y_land=np.linspace(1,16.36,64)
#y=np.linspace(1,13,50) #240*50+1000
f2,ax2 = plt.subplots(1, 1, figsize=(10.5,15))
pcm2 = ax2.pcolormesh(x_land, y_land, new_total_land)
#ax2.set_ylabel("Altitude", fontsize=14)
ax2.set_xlabel("Reflectivity [dBZ]", fontsize=28)
#ax.set_aspect((num_bins-num_removed_bins)/64)
ax2.set_aspect(135/64)
#ax2.set_title('Land', fontsize=14)
cb2=f2.colorbar(pcm2,ax=ax2, orientation = 'horizontal', fraction =0.049, pad=0.15)
cb2.set_label('Norm. occurrence (land)', fontsize=28)
ax2.tick_params(axis='both', which='major', labelsize=26)
cb2.set_ticks(np.arange(np.min(new_total_land), np.max(new_total_land), step=0.01))
cb2.ax.tick_params(labelsize=26, rotation=45)
#f2.tight_layout()
plt.savefig('Land_histogram_GAN_new')
#plt.show()

#Occurence difference histogram below
occurence_diff = new_total_ocean - new_total_land
x_diff=np.linspace(bin_edges_land[num_removed],20,num_bins-num_removed)
y_diff=np.linspace(1,16.36,64)
#y=np.linspace(1,13,50) #240*50+1000
f3,ax3 = plt.subplots(1, 1, figsize=(10.5,15))
pcm3 = ax3.pcolormesh(x_diff, y_diff, occurence_diff,cmap= 'seismic')
#ax3.set_ylabel("Altitude", fontsize=14)
ax3.set_xlabel("Reflectivity [dBZ]", fontsize=28)
#ax.set_aspect((num_bins-num_removed_bins)/64)
ax3.set_aspect(135/64)
#ax3.set_title('Occurence difference')
cb3=f3.colorbar(pcm3,ax=ax3, orientation = 'horizontal', fraction =0.049, pad=0.15)
cb3.set_label('Occurence bias (ocean - land)', fontsize=28)
ax3.tick_params(axis='both', which='major', labelsize=26)
#cb3.set_ticks(np.arange(-0.0075, 0.01, step=0.0025))
cb3.ax.tick_params(labelsize=26, rotation = 45)
#f3.tight_layout()
plt.savefig('Difference_histogram_OceanLandGAN_new')
#plt.show()
