import h5py
from IceWaterPathMethod import IceWaterPathMethod
from GAN_generator import GAN_generator
from plot_cloud import plot_cloud
from os import path
import torch
import numpy as np
import matplotlib.pyplot as plt
folder = './gan_training_results_ver_4/'
checkpoint_parameter = torch.load(folder + 'network_parameters.pt',map_location=torch.device('cpu'))
noise_parameter = checkpoint_parameter['noise_parameter']
print(noise_parameter)
H_gen=[384,16384, 256, 128, 64, 1]
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

b_size = 25
D_in_gen = [b_size, 64, 6]

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


xplot=np.linspace(0,64*1.1,64)
yplot=np.linspace(1,16.36,64)
from matplotlib.colors import Normalize

#f2, axtest = plt.subplots(1,1)
#ax=axs[range(0,3),range(0,3)]
norm = Normalize(-35, 20)

for k in range(0,5):
    f, axs = plt.subplots(5, 5, figsize=(12, 10))
    file_number = 0
    counter = 0
    noise = (torch.randn(D_in_gen)).to(device)
    output = netG(noise, None)
    output = (output + 1) * (55 / 2) - 35
    for i in range(0,5):
        for j in range(0,5):

            pcm = axs[i, j].pcolormesh(xplot, yplot, np.transpose(output.detach().numpy()[i + 5 * j][0]), norm=norm)
            title_str = 'Scene' + str(i)

            axs[i, j].tick_params(axis='both', which='major', labelsize='12')
            if i == 0:
                axs[j, i].set_ylabel('Generated \nAltitude [km]', fontsize=12)
            else:
                axs[j, i].tick_params(labelleft=False)
            if j == 4:
                axs[j, i].set_xlabel('Position [km]', fontsize=12)
            else:
                axs[j, i].tick_params(labelbottom=False)


            #with torch.no_grad():
            #    [IWP_generated_1, IWP_generated_2, IWP_generated_3] = IceWaterPathMethod(torch.transpose(output, 2, 3))
            #pcm = axs[0, i].semilogy(xplot, (IWP_generated_1[0, 0]), color=color_array[j], label='IWP generated', basey=10, alpha=0.5)
            #axs[6-(j+1), i].set_aspect(1.5)
            #cb.ax.tick_params(labelsize=2)


        #output = output.cpu().detach().numpy()[0, 0]


        #axs[0, i].set_aspect(1/4)
    f.tight_layout()
    f.subplots_adjust(right=0.88)
    cbar_ax = f.add_axes([0.9, 0.067, 0.025, 0.915])
    cbar1= f.colorbar(pcm, cax=cbar_ax)
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
    print('image saved as: ', 'testepoch' + str(epoch) + '_GAN_ver4_final'+str(k))
    plt.savefig('testepoch' + str(epoch) + '_GAN_ver4_final'+str(k))