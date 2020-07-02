import h5py

from GAN_generator import GAN_generator
from plot_cloud import plot_cloud
from os import path
import torch
import numpy as np
import matplotlib.pyplot as plt
folder = './training_results_cgan_ver1/'
checkpoint_parameter = torch.load(folder + 'network_parameters_CGAN.pt',map_location=torch.device('cpu'))
noise_parameter = checkpoint_parameter['noise_parameter']
print(noise_parameter)
H_gen=[704,16384, 256, 128, 64, 1]
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

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


counter = 0
for cloudsat_file in range(0, 500):
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

dataset = torch.utils.data.TensorDataset(cloudsat_scenes,modis_scenes)








xplot=range(0,64)
yplot=range(0,64)
from matplotlib.colors import Normalize
f,axs = plt.subplots(5,2)


#f2, axtest = plt.subplots(1,1)
#ax=axs[range(0,3),range(0,3)]
norm = Normalize(-35, 20)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True,
                                             num_workers=2)

for i, data in enumerate(dataloader,0):
    if i>4:
        break
    cs = data[0]
    print(cs.shape)
    string_ex = 'example' + str(i)
    modis = data[1]

    modis = torch.transpose(modis, 1, 3)
    modis = torch.transpose(modis, 2, 3)
    print(modis.shape)
    noise = (torch.randn(D_in_gen)*0.001).to(device)
    print(noise.shape)
    output = netG(noise, modis)
    output = (output + 1) * (55 / 2) - 35
    cs = (cs + 1) * (55 / 2) - 35
    print(output.detach().numpy().shape)

    pcm = axs[i,0].pcolormesh(xplot, yplot, (output.detach().numpy()[0,0]), norm=norm)
    title_str = 'Scene' + str(i)
    axs[i,0].set_title(title_str, fontsize=2)
    cb = f.colorbar(pcm, ax=axs[i,0])
    cb.set_label('Reflectivities [dBZ]', fontsize=2)
    axs[i,0].tick_params(axis='both', which='major', labelsize='2')
    cb.ax.tick_params(labelsize=2)

    pcm = axs[i, 1].pcolormesh(xplot, yplot, np.transpose(cs.detach().numpy()[0,0]), norm=norm)
    title_str = 'Scene' + str(i)
    axs[i, 1].set_title(title_str, fontsize=2)
    cb = f.colorbar(pcm, ax=axs[i, 1])
    cb.set_label('Reflectivities [dBZ]', fontsize=2)
    axs[i, 1].tick_params(axis='both', which='major', labelsize='2')
    cb.ax.tick_params(labelsize=2)

plt.savefig('testepoch' + str(epoch) + '_CGAN_ver1_1')
