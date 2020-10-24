import h5py
import torch
import  numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
from os import path
from GAN_generator import GAN_generator

def fmt(x,pos):
    a,b='{:.2e}'.format(x).split('e')
    b=int(b)
    return r'${} \times 10^{{{}}}$'.format(a,b)
    #return r'${} $'.format(a, b)


#hf = h5py.File('./modis_cloudsat_data/modis_cloudsat_ElevLatLong_test_data_conc_ver2.h5', 'r')
hf = h5py.File('CGAN_test_data_with_temp_conc_ver2.h5', 'r')
DEM_elevation = torch.tensor(hf.get('DEM_elevation'))
modis_scenes = torch.tensor((hf.get('modis_scenes'))).view(-1, 64, 10)
cloudsat_scenes = torch.tensor((hf.get('cloudsat_scenes'))).view(-1,64, 64)

cloudsat_scenes = (cloudsat_scenes + 1)*(55 / 2)-35
modis_scenes = torch.cat([modis_scenes[:,:,0:3],modis_scenes[:,:,4:9]],2)

print(DEM_elevation.shape)
print(modis_scenes.shape)
print(cloudsat_scenes.shape)

# Calculate distribution of original data set over ocean and land
'''
ocean_scene_counter = 0
land_scene_counter = 0
ocean_cloudsat = []
land_cloudsat = []
ocean_modis = []
land_modis = []


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
            ocean_modis = modis_scenes[i,:,:]
        else:
            ocean_cloudsat = np.append(ocean_cloudsat,cloudsat_scenes[i,:,:],0)
            ocean_modis = np.append(ocean_modis,modis_scenes[i,:,:],0)
    else:
        print('scene number ', i, ' over land')
        land_scene_counter = land_scene_counter +1
        if land_scene_counter == 1:
            land_cloudsat = cloudsat_scenes[i,:,:]
            land_modis = modis_scenes[i,:,:]
        else:
            land_cloudsat = np.append(land_cloudsat,cloudsat_scenes[i,:,:],0)
            land_modis = np.append(land_modis,modis_scenes[i,:,:],0)
print('Number of scenes over ocean', ocean_scene_counter)
print('Number of scenes over land', land_scene_counter)
print(ocean_cloudsat.shape)
print(ocean_modis.shape)
print(land_cloudsat.shape)
print(land_modis.shape)

hf = h5py.File('CGAN_test_ocean_land_conc.h5', 'w')
hf.create_dataset('ocean_cloudsat_CGAN', data=ocean_cloudsat)
hf.create_dataset('land_cloudsat_CGAN', data=land_cloudsat)
hf.create_dataset('ocean_modis_CGAN', data=ocean_modis)
hf.create_dataset('land_modis_CGAN', data=land_modis)
hf.close()
'''

# Load calculated distribution of original data set over ocean and land
hf = h5py.File('CGAN_test_ocean_land_conc.h5', 'r')
ocean_cloudsat = np.array(hf.get('ocean_cloudsat_CGAN'))
land_cloudsat = np.array(hf.get('land_cloudsat_CGAN'))
ocean_modis = np.array(hf.get('ocean_modis_CGAN'))
land_modis = np.array(hf.get('land_modis_CGAN'))

print(land_cloudsat.shape)
print(land_modis.shape)


#Histogram calculations for CloudSat and MODIS below
#region
num_bins = 55*4
num_bins_modis = 1000

num_removed = int((30/165)*num_bins)
total_ocean_histogram_cs = np.ones([64,num_bins])
total_land_histogram_cs = np.ones([64,num_bins])

total_ocean_histogram_modis = np.ones([8,num_bins_modis])
total_land_histogram_modis = np.ones([8,num_bins_modis])

#Ocean and land histograms below
dBz_per_bin = 55/num_bins
radiance_per_bin = 2/num_bins_modis
for i in range(0,64):
    ocean_scenes_histogram_cs, bin_edges_ocean_cs = np.histogram(ocean_cloudsat[:, i], bins=num_bins, range=(-35, 20), density=False)
    land_scenes_histogram_cs, bin_edges_land_cs = np.histogram(land_cloudsat[:, i], bins=num_bins, range=(-35, 20), density=False)
    sum_ocean_cs = np.sum(ocean_scenes_histogram_cs)
    sum_land_cs = np.sum(land_scenes_histogram_cs)

    total_ocean_histogram_cs[i] = ocean_scenes_histogram_cs/(sum_ocean_cs*dBz_per_bin)
    total_land_histogram_cs[i] = land_scenes_histogram_cs/(sum_land_cs*dBz_per_bin)
for i in range(0,8):
    ocean_scenes_histogram_modis, bin_edges_ocean_modis = np.histogram(ocean_modis[:, i], bins=num_bins_modis, range=(-1, 1), density=False) #maybe we should not be using the normed modis input but the original
    land_scenes_histogram_modis, bin_edges_land_modis = np.histogram(land_modis[:, i], bins=num_bins_modis, range=(-1, 1), density=False)
    sum_ocean_modis = np.sum(ocean_scenes_histogram_modis)
    sum_land_modis = np.sum(land_scenes_histogram_modis)
    total_ocean_histogram_modis[i] = ocean_scenes_histogram_modis/(sum_ocean_modis*radiance_per_bin)
    total_land_histogram_modis[i] = land_scenes_histogram_modis/(sum_land_modis*radiance_per_bin)

print('Ocean histogram shape cloudsat',total_ocean_histogram_cs.shape)
print('Land histogram shape cloudsat',total_land_histogram_cs.shape)
print('Ocean histogram shape modis',total_ocean_histogram_modis.shape)
print('Land histogram shape modis',total_land_histogram_modis.shape)


new_total_ocean_cs = total_ocean_histogram_cs[:,num_removed:num_bins]
print(new_total_ocean_cs.shape)
new_total_land_cs = total_land_histogram_cs[:,num_removed:num_bins]
#endregion


#Generate scenes with CGAN

#region
'''
land_cloudsat =torch.tensor(land_cloudsat.reshape(-1,64))
ocean_cloudsat =torch.tensor(ocean_cloudsat.reshape(-1,64))
land_modis =torch.tensor(land_modis.reshape(-1,1,64,8))
ocean_modis = torch.tensor(ocean_modis.reshape(-1,1,64,8))

ocean_modis = torch.transpose(ocean_modis, 1, 3)
ocean_modis = torch.transpose(ocean_modis, 2, 3)

land_modis = torch.transpose(land_modis, 1, 3)
land_modis = torch.transpose(land_modis, 2, 3)

#land_modis = land_modis[0:100,:,:,:]
#ocean_modis = ocean_modis[0:100,:,:,:]
#land_cloudsat = land_cloudsat[0:100*64,:]
#ocean_cloudsat = ocean_cloudsat[0:100*64,:]

print(land_cloudsat.shape)
print(land_modis.shape)
print(ocean_cloudsat.shape)
print(ocean_modis.shape)

D_gen_ocean = [len(ocean_cloudsat)//(64*100), 1,1,64]
D_gen_land = [len(land_cloudsat)//(64*100), 1,1,64]
H_gen = [576, 16384, 256, 128, 64, 1]
netG = GAN_generator(H_gen)
folder = './'
print('starting to generate scenes')
checkpoint_parameter = torch.load(folder + 'network_parameters_CGAN_3500.pt',map_location=torch.device('cpu'))
noise_parameter = checkpoint_parameter['noise_parameter']
print(noise_parameter)
real_label = 1
fake_label = 0
beta1 = 0.5
criterion = torch.nn.BCELoss()
lr = 0.0002
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
netG.load_state_dict(checkpoint_parameter['model_state_dict_gen'])

print('dadada')
all_generated_ocean = []
all_generated_land = []
for i in range(0, 100):
    print('Test shape ', (ocean_modis[i * len(ocean_cloudsat)//(64*100):(i)* len(ocean_cloudsat)//(64*100)+len(ocean_cloudsat)//(64*100), :, :, :]).shape)
    print((torch.randn(D_gen_ocean)).shape)
    generated_ocean = netG(torch.randn(D_gen_ocean), ocean_modis[i * len(ocean_cloudsat)//(64*100):(i)* len(ocean_cloudsat)//(64*100)+len(ocean_cloudsat)//(64*100), :, :, :])
    print(generated_ocean.shape, 'Generated ocean scene shape')
    generated_land = netG(torch.randn(D_gen_land), land_modis[i * len(land_cloudsat)//(64*100):(i)*len(land_cloudsat)//(64*100)+len(land_cloudsat)//(64*100), :, :, :])
    generated_ocean = torch.transpose(generated_ocean,2,3)
    generated_land = torch.transpose(generated_land, 2, 3)
    generated_ocean = generated_ocean.reshape(-1, 64)
    generated_land = generated_land.reshape(-1, 64)
    generated_ocean = generated_ocean.detach().numpy()
    generated_land = generated_land.detach().numpy()
    generated_ocean= np.array(generated_ocean)
    generated_land = np.array(generated_land)
    print(i)
    if i == 0:
        all_generated_ocean = generated_ocean
        all_generated_land = generated_land
    else:
        all_generated_ocean = np.append(all_generated_ocean, generated_ocean, axis=0)
        all_generated_land = np.append(all_generated_land, generated_land, axis=0)

all_generated_ocean = (all_generated_ocean + 1) * (55 / 2) - 35
all_generated_land = (all_generated_land + 1) * (55 / 2) - 35

print(all_generated_ocean.shape,' shape of all generated ocean' )
print(all_generated_land,' shape of all generated land')
hf = h5py.File('CGAN_test_ocean_land_generated_conc.h5', 'w')
hf.create_dataset('generated_ocean', data=all_generated_ocean)
hf.create_dataset('generated_land', data=all_generated_land)
hf.close()


#all_generated_ocean = all_generated_ocean.reshape(-1,64,64)
#from matplotlib.colors import Normalize
#f5,axs5 = plt.subplots(1,1)
#xplot = np.linspace(0, 64 * 1.1, 64)
#yplot = np.linspace(1, 16.36, 64)
#ax=axs[range(0,3),range(0,3)]
#norm = Normalize(-35, 20)
#pcm=axs5.pcolormesh(xplot,yplot, np.transpose(all_generated_ocean[19]), norm=norm)
#title_str = 'Scene' + str(4)
#axs5.set_title(title_str, fontsize=10)
#cb5 = f5.colorbar(pcm, ax=axs5)
#cb5.set_label('Reflectivites [dBZ]', fontsize=10)
#axs5.tick_params(axis='both', which='major', labelsize='10')
#cb5.ax.tick_params(labelsize=10)
#axs5.set_xlabel("Position [km]")
#axs5.set_ylabel("Altitude [km]")
#plt.show()


print('Shape of all generated ocean',all_generated_ocean.shape)
print('Shape of all generated land',all_generated_land.shape)
generated_land = all_generated_land
generated_ocean = all_generated_ocean
print('scenes generated')
'''
#endregion



# Load generated scenes in region below
#region
hf = h5py.File('CGAN_test_ocean_land_generated_conc.h5', 'r')
generated_ocean = np.array(hf.get('generated_ocean'))
generated_land = np.array(hf.get('generated_land'))
#endregion

#Histogram for generated scenes below
#region

total_ocean_histogram_gen = np.ones([64,num_bins])
total_land_histogram_gen = np.ones([64,num_bins])

for i in range(0,64):
    ocean_scenes_histogram_gen, bin_edges_ocean_gen = np.histogram(generated_ocean[:, i], bins=num_bins, range=(-35, 20), density=False)
    land_scenes_histogram_gen, bin_edges_land_gen = np.histogram(generated_land[:, i], bins=num_bins, range=(-35, 20), density=False)
    sum_ocean_gen = np.sum(ocean_scenes_histogram_gen)
    sum_land_gen = np.sum(land_scenes_histogram_gen)

    total_ocean_histogram_gen[i] = ocean_scenes_histogram_gen/(sum_ocean_gen*dBz_per_bin)
    total_land_histogram_gen[i] = land_scenes_histogram_gen/(sum_land_gen*dBz_per_bin)

new_total_ocean_gen = total_ocean_histogram_gen[:,num_removed:num_bins]
new_total_land_gen = total_land_histogram_gen[:,num_removed:num_bins]
occurence_diff_ocean = new_total_ocean_gen - new_total_ocean_cs
occurence_diff_land = new_total_land_gen - new_total_land_cs
#endregion

#Histogram plots for CloudSat below:
#region
x_ocean_cs=np.linspace(bin_edges_ocean_cs[num_removed],20,num_bins-num_removed)
y_ocean_cs=np.linspace(1,16.36,64)
f1,ax1 = plt.subplots(1, 1, figsize=(10.5,15))
pcm1 = ax1.pcolormesh(x_ocean_cs, y_ocean_cs, new_total_ocean_cs, vmin=0, vmax=4e-2)
ax1.set_ylabel("Altitude [km]", fontsize=28)
ax1.set_xlabel("Reflectivity [dBZ]", fontsize=28)
#ax1.set_aspect((num_bins-num_removed)/64)
ax1.set_aspect(155/64)
ax1.set_title('Ocean', fontsize=34)
#ax1.set_title('Ocean', fontsize=14)
cb1=f1.colorbar(pcm1,ax=ax1, orientation = 'horizontal', fraction =0.049, pad=0.15)
cb1.set_label('Norm. occurrence (Ocean) [dBZ$^{-1}$]', fontsize=28)
ax1.tick_params(axis='both', which='major', labelsize=26)
#cb1.set_ticks(np.arange(np.min(new_total_ocean_cs), np.max(new_total_ocean_cs), step=0.01))
cb1.ax.tick_params(labelsize=26, rotation=45)
#cb1.ax.tick_params(labelsize=12)
#f1.tight_layout()
plt.savefig('./Results/HistogramCGAN/Ocean_histogram_CGAN_Cloudsat.png')

x_land_cs=np.linspace(bin_edges_land_cs[num_removed],20,num_bins-num_removed)
y_land_cs=np.linspace(1,16.36,64)
#y=np.linspace(1,13,50) #240*50+1000
f2,ax2 = plt.subplots(1, 1, figsize=(10.5,15))
pcm2 = ax2.pcolormesh(x_land_cs, y_land_cs, new_total_land_cs, vmin=0, vmax=4e-2)
#ax2.set_ylabel("Altitude", fontsize=14)
ax2.set_xlabel("Reflectivity [dBZ]", fontsize=28)
#ax2.set_aspect((num_bins-num_removed)/64)
ax2.set_aspect(155/64)
ax2.set_title('Land', fontsize=34)
#ax2.set_title('Land', fontsize=14)
cb2=f2.colorbar(pcm2,ax=ax2, orientation = 'horizontal', fraction =0.049, pad=0.15)
cb2.set_label('Norm. occurrence (Land) [dBZ$^{-1}$]', fontsize=28)
ax2.tick_params(axis='both', which='major', labelsize=26)
#cb2.set_ticks(np.arange(np.min(new_total_land_cs), np.max(new_total_land_cs), step=0.01))
cb2.ax.tick_params(labelsize=26, rotation=45)
#f2.tight_layout()
plt.savefig('./Results/HistogramCGAN/Land_histogram_CGAN_Cloudsat.png')


#Occurence difference histogram below
occurence_diff_cs = new_total_ocean_cs - new_total_land_cs
x_diff_cs=np.linspace(bin_edges_land_cs[num_removed],20,num_bins-num_removed)
y_diff_cs=np.linspace(1,16.36,64)
#y=np.linspace(1,13,50) #240*50+1000
f3,ax3 = plt.subplots(1, 1, figsize=(10.5,15))
pcm3 = ax3.pcolormesh(x_diff_cs, y_diff_cs, occurence_diff_cs, vmin = -0.017, vmax = 0.017, cmap= 'seismic')
#ax3.set_ylabel("Altitude", fontsize=14)
ax3.set_xlabel("Reflectivity [dBZ]", fontsize=28)
#ax3.set_aspect((num_bins-num_removed)/64)
ax3.set_aspect(155/64)
ax3.set_title('Ocean - Land', fontsize=34)
#ax3.set_title('Occurrence difference')
cb3=f3.colorbar(pcm3,ax=ax3, orientation = 'horizontal', fraction =0.049, pad=0.15)
cb3.set_label('Occurrence bias (Ocean - Land) [dBZ$^{-1}$]', fontsize=28)
ax3.tick_params(axis='both', which='major', labelsize=26)
#cb3.set_ticks(np.arange(-0.0075, 0.01, step=0.0025))
cb3.ax.tick_params(labelsize=26, rotation = 45)
#f3.tight_layout()
plt.savefig('./Results/HistogramCGAN/Difference_histogram_OceanLandCGAN_Cloudsat.png')

#endregion

#Histogram plots for MODIS below:
#region

x_ocean_modis=bin_edges_ocean_modis
y_ocean_modis=[27,28,29,31,32,33,34,35]
f4,ax4 = plt.subplots(1, 1, figsize=(10.5,15))
pcm4 = ax4.pcolormesh(x_ocean_modis, y_ocean_modis, total_ocean_histogram_modis, vmin = 0, vmax=3)
ax4.set_ylabel("MODIS band", fontsize=28)
ax4.set_xlabel("Normalized radiance", fontsize=28)
#ax.set_aspect((num_bins-num_removed_bins)/64)
ax4.set_aspect(2/8.5)
#ax1.set_title('Ocean', fontsize=14)
cb4=f4.colorbar(pcm4,ax=ax4, orientation = 'horizontal', fraction =0.049, pad=0.15)
cb4.set_label('Norm. occurrence (Ocean MODIS)', fontsize=28)
ax4.tick_params(axis='both', which='major', labelsize=26)
#cb1.set_ticks(np.arange(np.min(new_total_ocean_cs), np.max(new_total_ocean_cs), step=0.01))
cb4.ax.tick_params(labelsize=26, rotation=45)
#cb1.ax.tick_params(labelsize=12)
#f1.tight_layout()
plt.savefig('./Results/HistogramCGAN/Ocean_histogram_CGAN_modis.png')

x_land_modis=bin_edges_land_modis
y_land_modis=[27,28,29,31,32,33,34,35]
f5,ax5 = plt.subplots(1, 1, figsize=(10.5,15))
pcm5 = ax5.pcolormesh(x_land_modis, y_land_modis, total_land_histogram_modis, vmin = 0, vmax=3)
ax5.set_xlabel("Normalized radiance", fontsize=28)
#ax.set_aspect((num_bins-num_removed_bins)/64)
ax5.set_aspect(2/8.5)
#ax2.set_title('Land', fontsize=14)
cb5=f5.colorbar(pcm5,ax=ax5, orientation = 'horizontal', fraction =0.049, pad=0.15)
cb5.set_label('Norm. occurrence (Land MODIS)', fontsize=28)
ax5.tick_params(axis='both', which='major', labelsize=26)
#cb2.set_ticks(np.arange(np.min(new_total_land_cs), np.max(new_total_land_cs), step=0.01))
cb5.ax.tick_params(labelsize=26, rotation=45)
#f2.tight_layout()
plt.savefig('./Results/HistogramCGAN/Land_histogram_CGAN_modis.png')


#Occurence difference histogram below
occurence_diff_modis = total_ocean_histogram_modis - total_land_histogram_modis
x_diff_modis=bin_edges_land_modis
y_diff_modis=[27,28,29,31,32,33,34,35]
f6,ax6 = plt.subplots(1, 1, figsize=(10.5,15))
pcm6 = ax6.pcolormesh(x_diff_modis, y_diff_modis, occurence_diff_modis, vmin=-0.4, vmax=0.4, cmap= 'seismic')
ax6.set_xlabel("Normalized radiance", fontsize=28)
#ax.set_aspect((num_bins-num_removed_bins)/64)
ax6.set_aspect(2/8.5)
#ax3.set_title('Occurence difference')
cb6=f6.colorbar(pcm6,ax=ax6, orientation = 'horizontal', fraction =0.049, pad=0.15)
cb6.set_label('Occurrence bias (Ocean - Land)', fontsize=28)
ax6.tick_params(axis='both', which='major', labelsize=26)
#cb3.set_ticks(np.arange(-0.0075, 0.01, step=0.0025))
cb6.ax.tick_params(labelsize=26, rotation = 45)
#f3.tight_layout()
plt.savefig('./Results/HistogramCGAN/Difference_histogram_OceanLandCGAN_modis.png')

#endregion

#Histogran plots for generated scenes
#region
x_ocean_cs=np.linspace(bin_edges_ocean_cs[num_removed],20,num_bins-num_removed)
y_ocean_cs=np.linspace(1,16.36,64)
f1,ax1 = plt.subplots(1, 1, figsize=(10.5,15))
pcm1 = ax1.pcolormesh(x_ocean_cs, y_ocean_cs, new_total_ocean_gen, vmin=0, vmax=4e-2)
ax1.set_ylabel("Altitude [km]", fontsize=28)
ax1.set_xlabel("Reflectivity [dBZ]", fontsize=28)
#ax1.set_aspect((num_bins-num_removed)/64)
ax1.set_aspect(155/64)
#ax1.set_title('Ocean', fontsize=14)
cb1=f1.colorbar(pcm1,ax=ax1, orientation = 'horizontal', fraction =0.049, pad=0.15)
cb1.set_label('Norm. occurrence (Ocean CGAN) [dBZ$^{-1}$]', fontsize=28)
ax1.tick_params(axis='both', which='major', labelsize=26)
#cb1.set_ticks(np.arange(np.min(new_total_ocean_cs), np.max(new_total_ocean_cs), step=0.01))
cb1.ax.tick_params(labelsize=26, rotation=45)
#cb1.ax.tick_params(labelsize=12)
#f1.tight_layout()
plt.savefig('./Results/HistogramCGAN/Ocean_histogram_CGAN_generated.png')

x_land_cs=np.linspace(bin_edges_land_cs[num_removed],20,num_bins-num_removed)
y_land_cs=np.linspace(1,16.36,64)
#y=np.linspace(1,13,50) #240*50+1000
f2,ax2 = plt.subplots(1, 1, figsize=(10.5,15))
pcm2 = ax2.pcolormesh(x_land_cs, y_land_cs, new_total_land_gen, vmin=0, vmax=4e-2)
#ax2.set_ylabel("Altitude", fontsize=14)
ax2.set_xlabel("Reflectivity [dBZ]", fontsize=28)
#ax2.set_aspect((num_bins-num_removed)/64)
ax2.set_aspect(155/64)
#ax2.set_title('Land', fontsize=14)
cb2=f2.colorbar(pcm2,ax=ax2, orientation = 'horizontal', fraction =0.049, pad=0.15)
cb2.set_label('Norm. occurrence (Land CGAN) [dBZ$^{-1}$]', fontsize=28)
ax2.tick_params(axis='both', which='major', labelsize=26)
#cb2.set_ticks(np.arange(np.min(new_total_land_cs), np.max(new_total_land_cs), step=0.01))
cb2.ax.tick_params(labelsize=26, rotation=45)
#f2.tight_layout()
plt.savefig('./Results/HistogramCGAN/Land_histogram_CGAN_generated.png')
#endregion

#Plots occurrence difference
#region
f3,ax3 = plt.subplots(1, 1, figsize=(10.5,15))
pcm3 = ax3.pcolormesh(x_land_cs, y_land_cs, occurence_diff_land, vmin = -0.017, vmax = 0.017, cmap= 'seismic')
#ax3.set_ylabel("Altitude", fontsize=28)
ax3.set_xlabel("Reflectivity [dBZ]", fontsize=28)
#ax.set_aspect((num_bins-num_removed_bins)/64)
ax3.set_aspect(155/64)
#ax3.set_title('Occurrence difference')
cb3=f3.colorbar(pcm3,ax=ax3, orientation = 'horizontal', fraction =0.049, pad=0.15)
cb3.set_label('Occurrence bias (CGAN - Land) [dBZ$^{-1}$]', fontsize=28)
ax3.tick_params(axis='both', which='major', labelsize=26)
#cb3.set_ticks(np.arange(-0.0075, 0.01, step=0.0025))
cb3.ax.tick_params(labelsize=26, rotation = 45)
#f3.tight_layout()
plt.savefig('./Results/HistogramCGAN/Difference_histogram_LandGenCGAN.png')

f3,ax3 = plt.subplots(1, 1, figsize=(10.5,15))
pcm3 = ax3.pcolormesh(x_ocean_cs, y_ocean_cs, occurence_diff_ocean, vmin = -0.017, vmax = 0.017, cmap= 'seismic')
ax3.set_ylabel("Altitude [km]", fontsize=28)
ax3.set_xlabel("Reflectivity [dBZ]", fontsize=28)
#ax.set_aspect((num_bins-num_removed_bins)/64)
ax3.set_aspect(155/64)
#ax3.set_title('Occurence difference')
cb3=f3.colorbar(pcm3,ax=ax3, orientation = 'horizontal', fraction =0.049, pad=0.15)
cb3.set_label('Occurrence bias (CGAN - Ocean) [dBZ$^{-1}$]', fontsize=28)
ax3.tick_params(axis='both', which='major', labelsize=26)
#cb3.set_ticks(np.arange(-0.0075, 0.01, step=0.0025))
cb3.ax.tick_params(labelsize=26, rotation = 45)
#f3.tight_layout()
plt.savefig('./Results/HistogramCGAN/Difference_histogram_OceanGenCGAN.png')
#endregion

#plt.show()

