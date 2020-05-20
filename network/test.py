
import os
import numpy.ma as ma
from datetime import datetime
import numpy as np
import torch

import matplotlib.pyplot as plt
from GAN_generator import GAN_generator
from GAN_discriminator import GAN_discriminator
from Create_Dataset import create_dataset
# parameters for the generator
H_gen=[16384, 256, 128, 64, 1]
D_in=384
D_out=[64,64]
N=15
num_epochs=1
num_batches = 1


netG = GAN_generator(D_in,H_gen,D_out)
x=torch.randn(D_in)
#generated scene

cloudsat_scenes= create_dataset()
scene_real = torch.from_numpy(cloudsat_scenes[110])
scene_real = scene_real.view(1,1,64,64)

H_disc=[256, 128, 128, 5, 6, 64, 128, 256, 256, 4096, 1]
D_in=[1, 5, 1,64]
D_out=[64,64]
N=15
scene=torch.randn([1,1,64,64])
netD = GAN_discriminator(D_in,H_disc,D_out)
x=torch.randn(D_in)
#probability that scene_gen is fake

b_size=1
real_label = 1
fake_label = 0
beta1=0.5
criterion = torch.nn.BCELoss()
lr=0.0002

optimizerD= torch.optim.Adam(netD.parameters(),lr=lr, betas = (beta1,0.999))
optimizerG= torch.optim.Adam(netG.parameters(),lr=lr, betas = (beta1,0.999))
