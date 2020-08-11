import h5py
from IceWaterPathMethod import IceWaterPathMethod
from GAN_generator import GAN_generator
from plot_cloud import plot_cloud
from os import path
import torch
import numpy as np
import matplotlib.pyplot as plt
folder = './'
checkpoint_parameter = torch.load(folder + 'network_parameters_CGAN.pt',map_location=torch.device('cpu'))
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





location = './modis_cloudsat_data/'
file_string = location + 'modis_cloudsat_test_data_conc_ver2' + '.h5'
hf = h5py.File(file_string, 'r')

cloudsat_scenes = torch.tensor(hf.get('cloudsat_scenes'))

modis_scenes = torch.tensor(hf.get('modis_scenes'))
print('modis scenes input size: ', modis_scenes.shape)
temp_modis_scenes = torch.cat([modis_scenes[:, :, :, 0:3], modis_scenes[:, :, :, 4:9]], 3)
modis_scenes = temp_modis_scenes
print('modis after change size: ', modis_scenes.shape)
dataset = torch.utils.data.TensorDataset(cloudsat_scenes, modis_scenes)

print('nuber of missing values in modis ', t_count(modis_scenes,(float(-1))))






xplot=np.linspace(0,64*1.1,64)
yplot=np.linspace(1,16.36,64)
from matplotlib.colors import Normalize



#f2, axtest = plt.subplots(1,1)
#ax=axs[range(0,3),range(0,3)]
norm = Normalize(-35, 20)
f, axs = plt.subplots(7,5, figsize=(13,15))
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
    print(modis.shape)
    modis = modis.to(device)
    pl = axs[1,i].plot(xplot,np.transpose(np.array(modis[0,:,0,:])))
    axs[1,i].set_ylim([-1,1])
    axs[1,i].minorticks_on()
    axs[1, i].tick_params(axis='both', which='major', labelsize='12')
    axs[1, i].tick_params(axis='both', which='minor')
    axs[1,i].tick_params(labelbottom=False)
    if i == 0:
        axs[1, i].set_ylabel('Modis Input', fontsize=12)
    else:
        axs[1, i].tick_params(labelleft=False)

    cs = (cs + 1) * (55 / 2) - 35
    [IWP_cs_1, IWP_cs_2, IWP_cs_3] = IceWaterPathMethod((cs))
    cs = cs.to(device)
    pcm1 = axs[6, i].pcolormesh(xplot, yplot, np.transpose(cs.cpu().detach().numpy()[0,0]), norm=norm)
    title_str = 'Scene' + str(i)
    #axs[6, i].set_title(title_str, fontsize=2)
    #cb = f.colorbar(pcm, ax=axs[counter, 0])
    #cb.set_label('Reflectivities [dBZ]', fontsize=2)
    axs[6, i].minorticks_on()
    axs[6,i].set_xlabel('Position [km]', fontsize=12)
    if i == 0:
        axs[6, i].set_ylabel('Real \nAltitude [km]', fontsize=12)
    else:
        axs[6, i].tick_params(labelleft=False)
    axs[6, i].tick_params(axis='both', which='both', labelsize='12')
    #axs[6, i].set_aspect(1.5)
    #cb.ax.tick_params(labelsize=2)
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
        axs[6-(j+1), i].tick_params(axis='both', which='major', labelsize='12')
        axs[6-(j+1), i].tick_params(axis='both', which='minor')
        axs[6-(j+1),i].tick_params(labelbottom=False)
        if i == 0:
            axs[6-(j+1), i].set_ylabel('Generated \nAltitude [km]', fontsize=12)
        else:
            axs[6 - (j + 1), i].tick_params(labelleft=False)
        #axs[6-(j+1), i].set_aspect(1.5)
        #cb.ax.tick_params(labelsize=2)


    #output = output.cpu().detach().numpy()[0, 0]

    with torch.no_grad():
        [IWP_generated_1, IWP_generated_2, IWP_generated_3] = IceWaterPathMethod(torch.transpose(output,2,3))
    axs[0, i].minorticks_on()
    axs[0, i].tick_params(axis='both', which='major', labelsize='12')
    axs[0, i].tick_params(axis='both', which='minor')
    axs[0,i].set_ylim([2**(-4),2**4])
    pcm = axs[0, i].semilogy(xplot, (IWP_generated_1[0, 0]), color='grey', label='IWP generated',basey=2)
    pcm = axs[0, i].semilogy(xplot, (IWP_cs_1[0,0]), color='black', label='IWP real',basey=2)
    title_str = 'IWP generated' + str(i)
    #axs[0,i].set_title(title_str, fontsize=2)
    axs[0, i].tick_params(labelbottom=False)
    if i == 0:
        axs[0, i].set_ylabel('IWP [g m⁻²]', fontsize=12)
    else:
        axs[0, i].tick_params(labelleft=False)
    #axs[0, i].set_aspect(1/4)

f.tight_layout()
f.subplots_adjust(right=0.88)
cbar_ax = f.add_axes([0.9, 0.045, 0.025, 0.665])
cbar1= f.colorbar(pcm1, cax=cbar_ax)
cbar1.set_label('Reflectivities [dBZ]', fontsize=12)
cbar1.ax.tick_params(labelsize=12)



'''

    pcm =axs[counter,2].plot(xplot, IWP_generated_1[0,0], label='IWP ver1')
    title_str = 'IWP generated' + str(i)
    axs[counter, 2].set_title(title_str, fontsize=2)

    pcm = axs[counter, 3].plot(xplot, IWP_cs_1[0,0], label='IWP ver1')
    title_str = 'IWP cloudsat' + str(i)
    axs[counter, 2].set_title(title_str, fontsize=2)
'''
print('image saved as: ', 'testepoch' + str(epoch) + '_CGAN_ver8_1')
plt.savefig('testepoch' + str(epoch) + '_CGAN_ver8_1')