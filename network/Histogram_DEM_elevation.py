
import h5py
import torch
import  numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from os import path
#from mpl_toolkits.axes_grid1 import make_axes_locatable

def fmt(x,pos):
    a,b='{:.2e}'.format(x).split('e')
    b=int(b)
    #return r'${} \times 10^{{{}}}$'.format(a,b)
    return r'${} $'.format(a, b)
'''
location = './GAN_elevation/test_data/'
counter = 0

hf = h5py.File('./rr_data/index_list.h5', 'r')
index_list = (np.array(hf.get('index_list')))
for ind in index_list:
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

hf = h5py.File(GAN_test_ocean_land_conc, 'w')
hf.create_dataset('ocean_cloudsat_GAN', data=ocean_cloudsat)
hf.create_dataset('land_cloudsat_GAN', data=land_cloudsat)
hf.close()
'''
#Load saved scenes over ocean/land from the real data set
hf = h5py.File('GAN_test_ocean_land_conc.h5','r')
ocean_cloudsat = torch.tensor(hf.get('ocean_cloudsat_GAN'))
land_cloudsat = torch.tensor(hf.get('land_cloudsat_GAN'))
print('loaded real ocean and land scenes')

hf2 = h5py.File('GAN_generated_scenes_for_histogram.h5','r')
generated_scenes = torch.tensor(hf2.get('all_generated')).view(-1,64,64)
print('loaded generated scenes')

print('Max gen scenes',torch.max(generated_scenes))
generated_scenes = generated_scenes.view(-1,64)

num_bins = 4*55
num_removed = int((30/165)*num_bins)
total_ocean_histogram = np.ones([64,num_bins])
total_land_histogram = np.ones([64,num_bins])
total_generated_histogram = np.ones([64,num_bins])

#Ocean and land histograms below
dBz_per_bin = 55/num_bins
for i in range(0,64):
    ocean_scenes_histogram, bin_edges_ocean = np.histogram(ocean_cloudsat[:, i], bins=num_bins, range=(-35, 20), density=False)
    land_scenes_histogram, bin_edges_land = np.histogram(land_cloudsat[:, i], bins=num_bins, range=(-35, 20), density=False)
    generated_scenes_histogram, bin_edges_generated = np.histogram(generated_scenes[:,i], bins=num_bins, range=(-35, 20), density=False)
    sum_ocean = np.sum(ocean_scenes_histogram)
    sum_land = np.sum(land_scenes_histogram)
    sum_generated = np.sum(generated_scenes_histogram)
    total_ocean_histogram[i] = ocean_scenes_histogram/(dBz_per_bin*sum_ocean)
    total_land_histogram[i] = land_scenes_histogram/(dBz_per_bin*sum_land)
    total_generated_histogram[i] = generated_scenes_histogram/(dBz_per_bin*sum_generated)
print('Ocean histogram shape ',total_ocean_histogram.shape)
print('Land histogram shape ',total_land_histogram.shape)

#sum_ocean = np.sum(total_ocean_histogram)
#sum_land = np.sum(total_land_histogram)
#sum_generated = np.sum(total_generated_histogram)



#total_ocean_histogram = total_ocean_histogram/(dBz_per_bin*sum_ocean)
#total_land_histogram = total_land_histogram/(dBz_per_bin*sum_land)
#total_generated_histogram = total_generated_histogram/(dBz_per_bin*sum_generated)

new_total_ocean = total_ocean_histogram[:,num_removed:num_bins]
print(new_total_ocean.shape)
new_total_land = total_land_histogram[:,num_removed:num_bins]
new_total_generated = total_generated_histogram[:,num_removed:num_bins]

#region
'''

occurence_diff = new_total_ocean - new_total_land

percentage_diff =np.zeros([64,len(new_total_ocean[0])])

zero_counter = 0
for i in range(0,64):
    for j in range(0,len(new_total_ocean[0])):
        if new_total_ocean[i,j] == 0:
            #print('i and j: ', i, j, 'Occ diff: ',occurence_diff[i,j])
            zero_counter=zero_counter+1
            percentage_diff[i,j] = np.nan
        else:
            percentage_diff[i,j] = occurence_diff[i,j]/new_total_ocean[i,j]
large_counter = 0
for i in range(0,64):
    for j in range(0,len(new_total_ocean[0])):
        if abs(percentage_diff[i,j]) > 2:
            #print('Large percentage diff at i:', i, ' j: ',j, ' Value: ', percentage_diff[i,j])
            large_counter = large_counter +1

print('Number of zeros in cloudsat: ',zero_counter)
print('Number of large percentages: ',large_counter)
print('percentage hist shape ',percentage_diff.shape)
#percentage_diff = percentage_diff *100

'''
#endregion

#Ocean/Land histograms below
x_ocean=np.linspace(bin_edges_ocean[num_removed],20,num_bins-num_removed)
y_ocean=np.linspace(1,16.36,64)
#y=np.linspace(1,13,50) #240*50+1000
f1,ax1 = plt.subplots(1, 1, figsize=(10.5,15))
pcm1 = ax1.pcolormesh(x_ocean, y_ocean, new_total_ocean, vmax = 0.035)
ax1.set_ylabel("Altitude [km]", fontsize=28)
ax1.set_xlabel("Reflectivity [dBZ]", fontsize=28)
#ax.set_aspect((num_bins-num_removed_bins)/64)
ax1.set_aspect(155/64)
ax1.set_title('Ocean', fontsize=34)
#ax1.set_title('Ocean', fontsize=14)
cb1=f1.colorbar(pcm1,ax=ax1, orientation = 'horizontal', fraction =0.049, pad=0.15)
cb1.set_label('Norm. occurrence (Ocean) [dBZ$^{-1}$]', fontsize=28)
ax1.tick_params(axis='both', which='major', labelsize=26)
#cb1.set_ticks(np.arange(np.min(new_total_ocean), np.max(new_total_ocean), step=0.01))
cb1.ax.tick_params(labelsize=26, rotation=45)
#cb1.ax.tick_params(labelsize=12)
#f1.tight_layout()
plt.savefig('./Results/HistogramGANepoch3500/Ocean_histogram_GAN_3500.png')

x_land=np.linspace(bin_edges_land[num_removed],20,num_bins-num_removed)
y_land=np.linspace(1,16.36,64)
#y=np.linspace(1,13,50) #240*50+1000
f2,ax2 = plt.subplots(1, 1, figsize=(10.5,15))
pcm2 = ax2.pcolormesh(x_land, y_land, new_total_land, vmax = 0.035)
#ax2.set_ylabel("Altitude", fontsize=14)
ax2.set_xlabel("Reflectivity [dBZ]", fontsize=28)
#ax.set_aspect((num_bins-num_removed_bins)/64)
ax2.set_aspect(155/64)
ax2.set_title('Land', fontsize=34)
#ax2.set_title('Land', fontsize=14)
cb2=f2.colorbar(pcm2,ax=ax2, orientation = 'horizontal', fraction =0.049, pad=0.15)
cb2.set_label('Norm. occurrence (Land) [dBZ$^{-1}$]', fontsize=28)
ax2.tick_params(axis='both', which='major', labelsize=26)
#cb2.set_ticks(np.arange(np.min(new_total_land), np.max(new_total_land), step=0.01))
cb2.ax.tick_params(labelsize=26, rotation=45)
#f2.tight_layout()
plt.savefig('./Results/HistogramGANepoch3500/Land_histogram_GAN_3500.png')


#Occurrence difference histogram below
occurence_diff = new_total_ocean - new_total_land
x_diff=np.linspace(bin_edges_land[num_removed],20,num_bins-num_removed)
y_diff=np.linspace(1,16.36,64)
#y=np.linspace(1,13,50) #240*50+1000
f3,ax3 = plt.subplots(1, 1, figsize=(10.5,15))
pcm3 = ax3.pcolormesh(x_diff, y_diff, occurence_diff, vmin = -0.017, vmax = 0.017, cmap= 'seismic')
#ax3.set_ylabel("Altitude", fontsize=14)
ax3.set_xlabel("Reflectivity [dBZ]", fontsize=28)
#ax.set_aspect((num_bins-num_removed_bins)/64)
ax3.set_aspect(155/64)
ax3.set_title('Ocean - Land', fontsize=34)
#ax3.set_title('Occurence difference')
cb3=f3.colorbar(pcm3,ax=ax3, orientation = 'horizontal', fraction =0.049, pad=0.15)
cb3.set_label('Occurrence bias (Ocean - Land) [dBZ$^{-1}$]', fontsize=28)
ax3.tick_params(axis='both', which='major', labelsize=26)
#cb3.set_ticks(np.arange(-0.0075, 0.01, step=0.0025))
cb3.ax.tick_params(labelsize=26, rotation = 45)
#f3.tight_layout()
plt.savefig('./Results/HistogramGANepoch3500/Difference_histogram_OceanLandGAN_3500.png')

#GAN generated histogram below
x_gen=np.linspace(bin_edges_generated[num_removed],20,num_bins-num_removed)
y_gen=np.linspace(1,16.36,64)
#y=np.linspace(1,13,50) #240*50+1000
f1,ax1 = plt.subplots(1, 1, figsize=(10.5,15))
pcm1 = ax1.pcolormesh(x_gen, y_gen, new_total_generated, vmax = 0.035)
ax1.set_ylabel("Altitude [km]", fontsize=28)
ax1.set_xlabel("Reflectivity [dBZ]", fontsize=28)
#ax.set_aspect((num_bins-num_removed_bins)/64)
ax1.set_aspect(155/64)
#ax1.set_title('Ocean', fontsize=14)
cb1=f1.colorbar(pcm1,ax=ax1, orientation = 'horizontal', fraction =0.049, pad=0.15)
cb1.set_label('Norm. occurrence (GAN) [dBZ$^{-1}$]', fontsize=28)
ax1.tick_params(axis='both', which='major', labelsize=26)
cb1.ax.tick_params(labelsize=26, rotation=45)
#cb1.ax.tick_params(labelsize=12)
#f1.tight_layout()
plt.savefig('./Results/HistogramGANepoch3500/Generated_histogram_GAN_3500.png')

occurence_diff_ocean_gan = new_total_generated - new_total_ocean
x_diff=np.linspace(bin_edges_land[num_removed],20,num_bins-num_removed)
y_diff=np.linspace(1,16.36,64)
#y=np.linspace(1,13,50) #240*50+1000
f3,ax3 = plt.subplots(1, 1, figsize=(10.5,15))
pcm3 = ax3.pcolormesh(x_diff, y_diff, occurence_diff_ocean_gan, vmin = -0.017, vmax = 0.017, cmap= 'seismic')
ax3.set_ylabel("Altitude [km]", fontsize=28)
ax3.set_xlabel("Reflectivity [dBZ]", fontsize=28)
#ax.set_aspect((num_bins-num_removed_bins)/64)
ax3.set_aspect(155/64)
#ax3.set_title('Occurence difference')
cb3=f3.colorbar(pcm3,ax=ax3, orientation = 'horizontal', fraction =0.049, pad=0.15)
cb3.set_label('Occurrence bias (GAN - Ocean) [dBZ$^{-1}$]', fontsize=28)
ax3.tick_params(axis='both', which='major', labelsize=26)
#cb3.set_ticks(np.arange(-0.0075, 0.01, step=0.0025))
cb3.ax.tick_params(labelsize=26, rotation = 45)
#f3.tight_layout()
plt.savefig('./Results/HistogramGAN/Difference_histogram_OceanGenGAN_new.png')

occurence_diff_land_gan = new_total_generated - new_total_land
x_diff=np.linspace(bin_edges_land[num_removed],20,num_bins-num_removed)
y_diff=np.linspace(1,16.36,64)
#y=np.linspace(1,13,50) #240*50+1000
f3,ax3 = plt.subplots(1, 1, figsize=(10.5,15))
pcm3 = ax3.pcolormesh(x_diff, y_diff, occurence_diff_land_gan, vmin = -0.017, vmax = 0.017, cmap= 'seismic')
#ax3.set_ylabel("Altitude", fontsize=28)
ax3.set_xlabel("Reflectivity [dBZ]", fontsize=28)
#ax.set_aspect((num_bins-num_removed_bins)/64)
ax3.set_aspect(155/64)
#ax3.set_title('Occurrence difference')
cb3=f3.colorbar(pcm3,ax=ax3, orientation = 'horizontal', fraction =0.049, pad=0.15)
cb3.set_label('Occurrence bias (GAN - Land) [dBZ$^{-1}$]', fontsize=28)
ax3.tick_params(axis='both', which='major', labelsize=26)
#cb3.set_ticks(np.arange(-0.0075, 0.01, step=0.0025))
cb3.ax.tick_params(labelsize=26, rotation = 45)
#f3.tight_layout()
plt.savefig('./Results/HistogramGAN/Difference_histogram_LandGenGAN_new.png')

'''
#Percentage difference histogram below
x_diff=np.linspace(bin_edges_land[num_removed],20,num_bins-num_removed)
y_diff=np.linspace(1,16.36,64)
#y=np.linspace(1,13,50) #240*50+1000
f3,ax3 = plt.subplots(1, 1, figsize=(10.5,15))
pcm3 = ax3.pcolormesh(x_diff, y_diff, percentage_diff,cmap= 'bone')
#ax3.set_ylabel("Altitude", fontsize=14)
ax3.set_xlabel("Reflectivity [dBZ]", fontsize=28)
#ax.set_aspect((num_bins-num_removed_bins)/64)
ax3.set_aspect(135/64)
#ax3.set_title('Occurence difference')
cb3=f3.colorbar(pcm3,ax=ax3, orientation = 'horizontal', fraction =0.049, pad=0.15)
cb3.set_label('Percentage occurence bias (Ocean - Land)/Ocean', fontsize=28)
ax3.tick_params(axis='both', which='major', labelsize=26)
#cb3.set_ticks(np.arange(-0.0075, 0.01, step=0.0025))
cb3.ax.tick_params(labelsize=26, rotation = 45)
#f3.tight_layout()
plt.savefig('./Results/HistogramGAN/Percentage_histogram_OceanLandGAN.png')
#plt.show()
'''
