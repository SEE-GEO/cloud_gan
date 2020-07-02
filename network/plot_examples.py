from GAN_generator import GAN_generator
from plot_cloud import plot_cloud
import torch
import numpy as np
import matplotlib.pyplot as plt
folder = './'
checkpoint_parameter = torch.load('network_parameters_2300.pt',map_location=torch.device('cpu'))
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
xplot=range(0,64)
yplot=range(0,64)
from matplotlib.colors import Normalize
f,axs = plt.subplots(5,5)

noise = ((torch.randn(D_in_gen))).to(device)
print(noise.shape)
output = netG(noise, None)
output = (output + 1) * (55 / 2) - 35

print(output.shape)
print(epoch)
#f2, axtest = plt.subplots(1,1)
#ax=axs[range(0,3),range(0,3)]
norm = Normalize(-35, 20)
for i in range(0,5):
    for j in range(0,5):
        pcm = axs[i, j].pcolormesh(xplot, yplot, np.transpose(output.detach().numpy()[i + 5*j][0]), norm=norm)
        title_str = 'Scene' + str(i)
        axs[i, j].set_title(title_str, fontsize=2)
        cb = f.colorbar(pcm, ax=axs[i, j])
        cb.set_label('Reflectivities [dBZ]', fontsize=2)
        axs[i, j].tick_params(axis='both', which='major', labelsize='2')
        cb.ax.tick_params(labelsize=2)

plt.savefig('testepoch'+ str(epoch) +'_GAN_ver3_1')