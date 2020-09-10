import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch


f, ax = plt.subplots(2,1)

ex_fontsize = 22
xplot=np.linspace(0,64*1.1,64)
yplot=np.linspace(1,16.36,64)
aspect = (64*1.1/(16.36-1))
print('Starting to load generated scenes')
location = './'
file_string = location + 'GAN_generated_scenes_for_histogram' + '.h5'
hf = h5py.File(file_string, 'r')

all_generated = torch.tensor(hf.get('all_generated'))
all_generated= all_generated.reshape(-1,64,64)
print('Generated scenes loaded')
print(all_generated.shape)
auto_corr_plot=np.zeros([len(all_generated), 64,64])
for j in range(0,len(all_generated)):
    for i in range(0,64):
        auto_corr_plot_temp = (np.correlate(all_generated[j, :, i], all_generated[j, :, i], mode='full'))
        auto_corr_plot_temp = auto_corr_plot_temp[auto_corr_plot_temp.size //2:] / max(auto_corr_plot_temp)
        auto_corr_plot[j,i,:] = auto_corr_plot_temp
        
    if j<5:
        f,ax = plt.subplots(2,1,figsize=(10,14))
        pcm1 = ax[0].pcolormesh(xplot,yplot,auto_corr_plot[j],vmin=-1,vmax=1, cmap='seismic')
        cbar1 = f.colorbar(pcm1, ax=ax[0],fraction =0.042, pad=0.04)
        cbar1.set_label('Autocorrelation ', fontsize=ex_fontsize)
        cbar1.ax.tick_params(labelsize=ex_fontsize)
        ax[0].set_aspect(aspect)
        pcm2 = ax[1].pcolormesh(xplot,yplot,np.transpose(all_generated[j, :, :]), vmin=-35, vmax=20)
        cbar2 = f.colorbar(pcm2, ax=ax[1],fraction =0.042, pad=0.04)
        cbar2.set_label('Reflectivity [dBZ] ', fontsize=ex_fontsize)
        cbar2.ax.tick_params(labelsize=ex_fontsize)
        ax[1].set_aspect(aspect)
        ax[1].tick_params(axis='both', which='major', labelsize=ex_fontsize)
        ax[0].tick_params(axis='both', which='major', labelsize=ex_fontsize)
        ax[1].set_xlabel('Position [km]', fontsize=ex_fontsize)
        ax[0].set_xlabel('Distance [km]', fontsize=ex_fontsize)
        ax[0].set_ylabel('Altitude [km]', fontsize=ex_fontsize)
        ax[1].set_ylabel('Altitude [km]', fontsize=ex_fontsize)
        f.tight_layout()
        plt.savefig('auto_corr_gan_' + str(j))
        #f.close()

f,ax = plt.subplots(1,1,figsize=(10.5,15))
average_auto_corr_gan = np.average(auto_corr_plot,0)
pcm1 = ax.pcolormesh(xplot,yplot,average_auto_corr_gan)
ax.tick_params(axis='both', which='major', labelsize=26)
ax.set_xlabel('Distance [km]', fontsize=28)
ax.set_ylabel('Altitude [km]', fontsize=28)
ax.set_aspect(aspect)
cbar1 = f.colorbar(pcm1, orientation = 'horizontal', fraction =0.049, pad=0.15)
cbar1.set_label('Autocorrelation (GAN)', fontsize=28)
cbar1.ax.tick_params(labelsize=26, rotation=45)
plt.savefig('auto_corr_gan_average')
#f.close()






print('Starting to load generated scenes')
location = './'
file_string = location + 'IAAFT_generated_scenes_GAN_testset' + '.h5'
hf = h5py.File(file_string, 'r')

iaaft_scenes = torch.tensor(hf.get('iaaft_scenes'))
iaaft_scenes= iaaft_scenes.reshape(-1,64,64)
print(torch.max(iaaft_scenes[0,:,:]))
num_files = len(iaaft_scenes)
print('Generated scenes loaded')
print(iaaft_scenes.shape)
iaaft_scenes = (iaaft_scenes + 1) * (55 / 2) - 35
auto_corr_plot=np.zeros([len(iaaft_scenes),64,64])

for j in range(0,len(iaaft_scenes)):
    for i in range(0,64):
        auto_corr_plot_temp = (np.correlate(iaaft_scenes[j, :, i], iaaft_scenes[j, :, i], mode='full'))
        auto_corr_plot_temp = auto_corr_plot_temp[auto_corr_plot_temp.size //2:] / max(auto_corr_plot_temp)
        auto_corr_plot[j,i,:] = auto_corr_plot_temp

    if j < 5:
        f, ax = plt.subplots(2, 1,figsize=(10,14))
        pcm1 = ax[0].pcolormesh(xplot,yplot,auto_corr_plot[j],vmin=-1,vmax=1,cmap='seismic')
        cbar1 = f.colorbar(pcm1, ax=ax[0],fraction =0.042, pad=0.04)
        cbar1.set_label('Autocorrelation ', fontsize=ex_fontsize)
        cbar1.ax.tick_params(labelsize=ex_fontsize)
        ax[0].set_aspect(aspect)
        pcm2 = ax[1].pcolormesh(xplot,yplot,np.transpose(iaaft_scenes[j, :, :]), vmin=-35, vmax=20)
        cbar2 = f.colorbar(pcm2, ax=ax[1],fraction =0.042, pad=0.04)
        cbar2.set_label('Reflectivity [dBZ] ', fontsize=ex_fontsize)
        cbar2.ax.tick_params(labelsize=ex_fontsize)
        ax[1].set_aspect(aspect)
        ax[1].tick_params(axis='both', which='major', labelsize=ex_fontsize)
        ax[0].tick_params(axis='both', which='major', labelsize=ex_fontsize)
        ax[1].set_xlabel('Position [km]', fontsize=ex_fontsize)
        ax[0].set_xlabel('Distance [km]', fontsize=ex_fontsize)
        ax[0].set_ylabel('Altitude [km]', fontsize=ex_fontsize)
        ax[1].set_ylabel('Altitude [km]', fontsize=ex_fontsize)
        f.tight_layout()
        plt.savefig('auto_corr_iaaft' + str(j))
        #f.close()
f,ax = plt.subplots(1,1,figsize=(10.5,15))
average_auto_corr_iaaft = np.average(auto_corr_plot,0)
ax.set_xlabel('Distance [km]', fontsize=28)
ax.tick_params(axis='both', which='major', labelsize=26)
ax.set_aspect(aspect)
pcm1 = ax.pcolormesh(xplot,yplot,average_auto_corr_iaaft)
cbar1 = f.colorbar(pcm1, orientation = 'horizontal', fraction =0.049, pad=0.15)
cbar1.set_label('Autocorrelation (IAAFT)', fontsize=28)
cbar1.ax.tick_params(labelsize=26, rotation=45)
plt.savefig('auto_corr_iaaft_average')
#f.close()


print('Starting to load generated scenes')
location = './rr_data/'
file_string = location + 'cloudsat_test_data_conc' + '.h5'
hf = h5py.File(file_string, 'r')

cs_scenes = torch.tensor(hf.get('cloudsat_scenes'))
cs_scenes = cs_scenes.reshape(-1, 64, 64)
cs_scenes = (cs_scenes + 1) * (55 / 2) - 35
print('Generated scenes loaded')
print(cs_scenes.shape)
print(torch.max(cs_scenes[0,:,:]))
auto_corr_plot=np.zeros([len(cs_scenes),64,64])

for j in range(0, len(cs_scenes)):
    for i in range(0, 64):
        auto_corr_plot_temp = (np.correlate(cs_scenes[j, :, i], cs_scenes[j, :, i], mode='full'))
        auto_corr_plot_temp = auto_corr_plot_temp[auto_corr_plot_temp.size // 2:] / max(auto_corr_plot_temp)
        auto_corr_plot[j, i, :] = auto_corr_plot_temp

    if j < 5:
        f, ax = plt.subplots(2, 1,figsize=(10,14))
        pcm1 = ax[0].pcolormesh(xplot,yplot,auto_corr_plot[j], cmap='seismic',vmin=-1,vmax=1)
        cbar1 = f.colorbar(pcm1, ax=ax[0],fraction =0.042, pad=0.04)
        cbar1.set_label('Autocorrelation ', fontsize=ex_fontsize)
        cbar1.ax.tick_params(labelsize=ex_fontsize)
        ax[0].set_aspect(aspect)
        pcm2 = ax[1].pcolormesh(xplot,yplot,np.transpose(cs_scenes[j, :, :]), vmin=-35, vmax=20)
        cbar2 = f.colorbar(pcm2, ax=ax[1],fraction =0.042, pad=0.04)
        cbar2.set_label('Reflectivity [dBZ] ', fontsize=ex_fontsize)
        cbar2.ax.tick_params(labelsize=ex_fontsize)
        ax[1].set_aspect(aspect)
        ax[1].tick_params(axis='both', which='major', labelsize=ex_fontsize)
        ax[0].tick_params(axis='both', which='major', labelsize=ex_fontsize)
        ax[1].set_xlabel('Position [km]', fontsize=ex_fontsize)
        ax[0].set_xlabel('Distance [km]', fontsize=ex_fontsize)
        ax[0].set_ylabel('Altitude [km]', fontsize=ex_fontsize)
        ax[1].set_ylabel('Altitude [km]', fontsize=ex_fontsize)
        f.tight_layout()
        plt.savefig('auto_corr_cs' + str(j))
        #f.close()

f,ax = plt.subplots(1,1,figsize=(10.5,15))
average_auto_corr_cs = np.average(auto_corr_plot,0)
print(np.max(average_auto_corr_cs))
pcm1 = ax.pcolormesh(xplot,yplot,average_auto_corr_cs)
ax.tick_params(axis='both', which='major', labelsize=26)
ax.set_xlabel('Distance [km]', fontsize=28)
ax.set_ylabel('Altitude [km]', fontsize=28)
ax.set_title('GAN', fontsize=32)
ax.set_aspect(aspect)
cbar1 = f.colorbar(pcm1, orientation = 'horizontal', fraction =0.049, pad=0.15)
cbar1.set_label('Autocorrelation (Real)', fontsize=28)
cbar1.ax.tick_params(labelsize=26, rotation=45)
plt.savefig('auto_corr_cs_average_for_gan')
#f.close()



f,ax = plt.subplots(1,1,figsize=(10.5,15))
average_auto_corr_cs = np.average(auto_corr_plot,0)
print(np.max(average_auto_corr_cs))
pcm1 = ax.pcolormesh(xplot,yplot,average_auto_corr_cs)
ax.set_xlabel('Distance [km]', fontsize=28)
ax.tick_params(axis='both', which='major', labelsize=26)
ax.set_title('IAAFT', fontsize=32)
ax.set_aspect(aspect)
cbar1 = f.colorbar(pcm1, orientation = 'horizontal', fraction =0.049, pad=0.15)
cbar1.set_label('Autocorrelation (Real) ', fontsize=28)
cbar1.ax.tick_params(labelsize=26, rotation=45)
plt.savefig('auto_corr_cs_average_for_iaaft')
#f.close()

f,ax = plt.subplots(1,1,figsize=(10.5,15))

pcm1 = ax.pcolormesh(xplot,yplot,average_auto_corr_iaaft - average_auto_corr_cs, cmap='seismic',vmin=-0.08,vmax=0.08)
ax.set_xlabel('Distance [km]', fontsize=28)
ax.tick_params(axis='both', which='major', labelsize=26)
ax.set_aspect(aspect)
cbar1 = f.colorbar(pcm1, orientation = 'horizontal', fraction =0.049, pad=0.15)
cbar1.set_label('Autocorrelation (IAAFT - Real)', fontsize=28)
cbar1.ax.tick_params(labelsize=26, rotation=45)
plt.savefig('auto_corr_cs_iaaft_difference_average')
#f.close()


f,ax = plt.subplots(1,1,figsize=(10.5,15))

pcm1 = ax.pcolormesh(xplot,yplot,average_auto_corr_gan - average_auto_corr_cs , cmap='seismic',vmin=-0.08,vmax=0.08)
ax.set_xlabel('Distance [km]', fontsize=28)
ax.set_ylabel('Altitude [km]', fontsize=28)
ax.set_aspect(aspect)
ax.tick_params(axis='both', which='major', labelsize=26)
cbar1 = f.colorbar(pcm1, orientation = 'horizontal', fraction =0.049, pad=0.15)
cbar1.set_label('Autocorrelation (GAN - Real)', fontsize=28)
cbar1.ax.tick_params(labelsize=26, rotation=45)
plt.savefig('auto_corr_cs_gan_difference_average')


