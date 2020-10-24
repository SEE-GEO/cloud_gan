import h5py
from IceWaterPathMethod import IceWaterPathMethod
from GAN_generator import GAN_generator
from plot_cloud import plot_cloud
from os import path
import torch
import numpy as np
import matplotlib.pyplot as plt
folder = './'
checkpoint_parameter = torch.load(folder + 'network_parameters_CGAN_3500.pt',map_location=torch.device('cpu'))
noise_parameter = checkpoint_parameter['noise_parameter']
print(noise_parameter)
H_gen=[576,16384, 256, 128, 64, 1]
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
epoch = checkpoint_parameter['epoch']
real_label = 1
fake_label = 0
beta1 = 0.5
criterion = torch.nn.BCELoss()
lr = 0.0002

f_size = 14

netG = GAN_generator(H_gen).float().to(device)
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
netG.load_state_dict(checkpoint_parameter['model_state_dict_gen'])

b_size = 1
D_in_gen = [b_size, 1,1,64]

'''
counter = 0
for cloudsat_file in range(0, 4900):
    location = './modis_cloudsat_data/test_data/'
    file_string = location + 'rr_modis_cloudsat_data_2015_' + str(cloudsat_file).zfill(4) + '.h5'
    if path.exists(file_string):
        hf = h5py.File(file_string, 'r')
        cloudsat_scenes_temp = torch.tensor(hf.get('rr')).view(-1, 1, 64, 64)
        modis_scenes_temp = torch.tensor(hf.get('emissivity')).view(-1, 1, 64, 10).float()
        if counter == 0:
            cloudsat_scenes = cloudsat_scenes_temp
            modis_scenes = modis_scenes_temp
            counter = 1
        else:
            cloudsat_scenes = torch.cat([cloudsat_scenes, cloudsat_scenes_temp], 0)
            modis_scenes = torch.cat([modis_scenes, modis_scenes_temp], 0)
temp_modis_scenes = torch.cat([modis_scenes[:,:,:,0:3],modis_scenes[:,:,:,4:9]],3)
modis_scenes=temp_modis_scenes
'''




def t_count(t, val):
    elements_equal_to_value = torch.eq(t, val)
    as_ints = elements_equal_to_value.numpy()
    count = np.sum(as_ints)
    return count





location = './modis_cloudsat_data/2016/'
#file_string = location + 'CGAN_test_data_with_temp_conc_ver2.h5' #for CGAN 2015
file_string = location + 'modis_cloudsat_ElevLatLong_test_data_conc_2016_normed2015_ver2.h5' #for CGAN 2016
hf = h5py.File(file_string, 'r')

cloudsat_scenes = torch.tensor(hf.get('cloudsat_scenes'))

modis_scenes = torch.tensor(hf.get('modis_scenes'))
print('modis scenes input size: ', modis_scenes.shape)
temp_modis_scenes = torch.cat([modis_scenes[:, :, :, 0:3], modis_scenes[:, :, :, 4:9]], 3)
modis_scenes = temp_modis_scenes
print('modis after change size: ', modis_scenes.shape)
dataset = torch.utils.data.TensorDataset(cloudsat_scenes, modis_scenes)

print('number of missing values in modis ', t_count(modis_scenes,(float(-1))))



xplot=np.linspace(0,64*1.1,64)
yplot=np.linspace(1,16.36,64)
from matplotlib.colors import Normalize


#f2, axtest = plt.subplots(1,1)
#ax=axs[range(0,3),range(0,3)]
norm = Normalize(-35, 20)
f, axs = plt.subplots(7,5, figsize=(13,18))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True,
                                             num_workers=2)
file_number = 0
counter = 0
for i, data in enumerate(dataloader,0):
    if i > 4:
        break
    cs = data[0]

    print('cs first element: ',cs[0,0,0,0])
    string_ex = 'example' + str(i)
    modis = data[1]

    modis = torch.transpose(modis, 1, 3)
    modis = torch.transpose(modis, 2, 3)
    print('modis shape', modis.shape)
    modis = modis.to(device)
    pl = axs[1,i].plot(xplot,np.transpose(np.array(modis[0,:,0,:])))
    axs[1,i].set_ylim([-1,1])
    axs[1,i].minorticks_on()
    axs[1, i].tick_params(axis='both', which='major', labelsize=f_size)
    axs[1, i].tick_params(axis='both', which='minor')
    axs[1,i].tick_params(labelbottom=False)
    if i == 0:
        axs[1, i].set_ylabel('Modis Input', fontsize=f_size)
    else:
        axs[1, i].tick_params(labelleft=False)

    cs = (cs + 1) * (55 / 2) - 35

    # Calculate cloud top height:
    altitudes_cs = np.zeros([1, 64])
    for k in range(0, 64):
        first_cloud_location_cs = 0
        for j in range(63, -1, -1):  # Loop backwards, starting from top of scene
            if cs[0,0, k, j] >= -15 and first_cloud_location_cs == 0:  # Set dBZ limit for cloud top detection
                altitudes_cs[0, k] = j  # save the index of the altitude where the cloud top is for each position
                first_cloud_location_cs = +1
    altitudes_cs = (altitudes_cs * 0.24) + 1  # Altitude of cloud top over sea level, [km]
    #print('Altitudes of cloud tops (CloudSat):', altitudes_template, ' [km]')

    cs = cs.to(device)
    print('cloudsat shape ', cs.shape)
    pcm1 = axs[6, i].pcolormesh(xplot, yplot, np.transpose(cs.cpu().detach().numpy()[0,0]), norm=norm)
    title_str = 'Scene' + str(i)
    #axs[7, i].set_title(title_str, fontsize=2)
    #cb = f.colorbar(pcm, ax=axs[counter, 0])
    #cb.set_label('Reflectivities [dBZ]', fontsize=2)
    axs[6, i].minorticks_on()
    axs[6,i].set_xlabel('Position [km]', fontsize=f_size)
    if i == 0:
        axs[6, i].set_ylabel('Real \nAltitude [km]', fontsize=f_size)
    else:
        axs[6, i].tick_params(labelleft=False)
    axs[6, i].tick_params(axis='both', which='both', labelsize=f_size)
    #axs[6, i].set_aspect(1.5)
    #cb.ax.tick_params(labelsize=2)

    color_array = ['blue','green','red','purple']
    for j in range(0,4):
        noise = (torch.randn(D_in_gen)).to(device)
        output = netG(noise, modis)
        output = (output + 1) * (55 / 2) - 35

        pcm = axs[6-(j+1), i].pcolormesh(xplot, yplot, output.detach().numpy()[0,0], norm=norm)
        title_str = 'Scene' + str(i)
        #axs[6-(j+1), i].set_title(title_str, fontsize=2)
        #cb = f.colorbar(pcm, ax=axs[counter, 0])
        #cb.set_label('Reflectivities [dBZ]', fontsize=2)
        axs[6-(j+1), i].minorticks_on()
        axs[6-(j+1), i].tick_params(axis='both', which='major', labelsize=f_size)
        axs[6-(j+1), i].tick_params(axis='both', which='minor')
        axs[6-(j+1),i].tick_params(labelbottom=False)
        if i == 0:
            axs[6-(j+1), i].set_ylabel('Generated \nAltitude [km]', fontsize=f_size)
        else:
            axs[6 - (j + 1), i].tick_params(labelleft=False)

        #axs[6-(j+1), i].set_aspect(1.5)
        #cb.ax.tick_params(labelsize=2)
        print('generated scenes shape: ',output.shape)
        # Calculate cloud top height:
        altitudes_generated = np.zeros([1, 64])
        for k in range(0, 64):
            first_cloud_location_gen = 0
            for l in range(63, -1, -1):  # Loop backwards, starting from top of scene
                if output[0,0,l,k] >= -15 and first_cloud_location_gen == 0:  # Set dBZ limit for cloud top detection
                    altitudes_generated[0, k] = l  # save the index of the altitude where the cloud top is for each position
                    first_cloud_location_gen = +1
        altitudes_generated = (altitudes_generated * 0.24) + 1  # Altitude of cloud top over sea level, [km]

        alt = axs[0,i].plot(xplot, np.transpose(altitudes_generated), color = color_array[j], label ='CTH generated',alpha=0.3)
    alt_cs = axs[0,i].plot(xplot, np.transpose(altitudes_cs),color='black', label='CTH real')
    axs[0, i].minorticks_on()
    axs[0, i].tick_params(axis='both', which='major', labelsize=f_size)
    axs[0, i].tick_params(axis='both', which='minor')
    axs[0, i].tick_params(labelbottom=False)
    if i == 0:
        axs[0, i].set_ylabel('Cloud-Top \nHeight [km]', fontsize=f_size)
    else:
        axs[0, i].tick_params(labelleft=False)
    #output = output.cpu().detach().numpy()[0, 0]
    axs[0, i].set_ylim(1, 16.36)


f.tight_layout()
f.subplots_adjust(right=0.88)
cbar_ax = f.add_axes([0.9, 0.0375, 0.025, 0.678]) #([distance from left, bottom height, width, height])
cbar1= f.colorbar(pcm1, cax=cbar_ax)
cbar1.set_label('Reflectivities [dBZ]', fontsize=f_size)
cbar1.ax.tick_params(labelsize=f_size)



'''
    pcm =axs[counter,2].plot(xplot, IWP_generated_1[0,0], label='IWP ver1')
    title_str = 'IWP generated' + str(i)
    axs[counter, 2].set_title(title_str, fontsize=2)
    pcm = axs[counter, 3].plot(xplot, IWP_cs_1[0,0], label='IWP ver1')
    title_str = 'IWP cloudsat' + str(i)
    axs[counter, 2].set_title(title_str, fontsize=2)
'''
print('image saved as: ', 'testepoch' + str(epoch) + '_CGAN_2016_example')
#plt.savefig('testepoch' + str(epoch) + '_CGAN_ver8_1')
#plt.savefig('./Results/2016//testepoch' + str(epoch) + '_CGAN_2016'+'_example4')
# Example1 normed with 2016 data set, the rest are normed with 2015 data set

