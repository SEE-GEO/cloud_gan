
import os
import os.path
from os import path
import h5py
import numpy.ma as ma
from datetime import datetime
import numpy as np
import torch

import matplotlib.pyplot as plt
from GAN_generator import GAN_generator
from GAN_discriminator import GAN_discriminator
#from Create_Dataset import create_dataset
# parameters for the generator
from plot_cloud import plot_cloud
def Training_GAN():

    H_gen=[384,16384, 256, 128, 64, 1]
    D_in_gen=[64,64,6]
    D_out=[64,64]
    N=15
    num_epochs=10
    num_files = 10
    num_batches = 1
    #ngpu=0
    batch_size=64
    workers=2

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    print(device)

    # for CGAN
    #H_disc = [5, 256, 128, 128, 5, 6, 64, 128, 256, 256, 4*4*256, 1]
    #for GAN

    H_disc = [5, 256, 128, 128, 5, 1, 64, 128, 256, 256, 4096, 1]
    netG = GAN_generator(H_gen).to(device)
    netD = GAN_discriminator(H_disc).to(device)

    #b_size=1
    real_label = 1
    fake_label = 0
    beta1=0.5
    criterion = torch.nn.BCELoss()
    lr=0.0002


    optimizerD= torch.optim.Adam(netD.parameters(),lr=lr, betas = (beta1,0.999))
    optimizerG= torch.optim.Adam(netG.parameters(),lr=lr, betas = (beta1,0.999))
    if path.exists('network_parameters.pt'):
        checkpoint = torch.load('network_parameters.pt')
        netG.load_state_dict(checkpoint['model_state_dict_gen'])
        optimizerG.load_state_dict(checkpoint['optimizer_state_dict_gen'])
        netD.load_state_dict(checkpoint['model_state_dict_disc'])
        optimizerD.load_state_dict(checkpoint['optimizer_state_dict_disc'])
        epoch_saved = checkpoint['epoch']
        G_losses = checkpoint['loss_gen']
        D_losses = checkpoint['loss_disc']
        noise_parameter = checkpoint['noise_parameter']

    else:
        G_losses=[]
        D_losses=[]
        epoch_saved=-1
        noise_parameter = 0.7
    #for each epoch

    for cloudsat_file in range(0,4998):
        location = './rr_data/'
        file_string = location + 'rr_data_2015_' + str(cloudsat_file).zfill(4) +'.h5'
        hf = h5py.File(file_string, 'r')

        cloudsat_scenes_temp = torch.tensor(hf.get('rr')).view(-1,1,64,64)
        if cloudsat_file == 0 :
            cloudsat_scenes=cloudsat_scenes_temp
        else:
            cloudsat_scenes = torch.cat([cloudsat_scenes,cloudsat_scenes_temp],0)





    for epoch in range(epoch_saved+1, 30):
        #for each batch
        dataloader = torch.utils.data.DataLoader(cloudsat_scenes, batch_size=batch_size, shuffle=True,
                                                 num_workers=workers)
        j=0
        for i, data in enumerate(dataloader,0):
            trainable =True
            #training discriminator with real data
            netD.zero_grad()
            real_cpu0 = data.to(device)
            b_size = real_cpu0.size(0)
            print(i)


            label=torch.full((b_size, ),real_label,device=device)
            #for CGAN
            #D_in_disc = [b_size, 5, 1, 64]
            #for GAN
            D_in_disc = [b_size, 1, 64, 64]
            D_out = [b_size,64, 64]

            noise = noise_parameter * torch.randn(D_in_disc)
            real_cpu1 = noise.to(device)


            output=netD(None,real_cpu0+real_cpu1).view(-1)

            errD_real = criterion(output,label)

            errD_real.backward()
            D_x = output.mean().item()

            #training discriminator with generated data
            D_in_gen = [b_size, 64, 6]
            noise = torch.randn(D_in_gen).to(device)

            fake = netG(noise)
            label.fill_(fake_label)

            noise = noise_parameter * torch.randn(D_in_disc)
            real_cpu1 = noise.to(device)

            output = netD(None,fake.detach() + real_cpu1).view(-1)

            errD_fake = criterion(output,label)

            D_G_z1 = output.mean().item()

            errD = errD_fake + errD_real
            if i%1!=0:
                trainable = False
            if trainable:
                errD_fake.backward()
                optimizerD.step()

            # update generator network
            netG.zero_grad()
            label.fill_(real_label) # fake labels are real for generator cost

            noise = noise_parameter * torch.randn(D_in_disc)
            real_cpu1 = noise.to(device)

            output = netD(None,fake + real_cpu1).view(-1)
            errG = criterion(output,label)

            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            j=j+b_size
            if i % 1 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, j, len(cloudsat_scenes),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
        noise_parameter = noise_parameter*0.85
        torch.save({
            'epoch': epoch,
            'model_state_dict_gen': netG.state_dict(),
            'optimizer_state_dict_gen': optimizerG.state_dict(),
            'loss_gen': G_losses,
            'model_state_dict_disc': netD.state_dict(),
            'optimizer_state_dict_disc': optimizerD.state_dict(),
            'loss_disc': D_losses,
            'noise_parameter' : noise_parameter
        }, 'network_parameters.pt')
        example_string = 'example_epoch_' + str(epoch).zfill(3)
        noise = torch.randn(D_in_gen).to(device)
        output0 = netG(noise)
        noise = torch.randn(D_in_gen).to(device)
        output1 = netG(noise)
        noise = torch.randn(D_in_gen).to(device)
        output2 = netG(noise)
        noise = torch.randn(D_in_gen).to(device)
        output3 = netG(noise)
        noise = torch.randn(D_in_gen).to(device)
        output4 = netG(noise)
        torch.save({
            'example0': output0,
            'example1': output1,
            'example2': output2,
            'example3': output3,
            'example4': output4
        }, example_string)
    print('done')
    #noise=torch.randn(D_in_gen)
    #output=netG(noise)
    #output=output.detach().numpy()
    #output=np.array(output)
    #plot_cloud(output[0][0])


