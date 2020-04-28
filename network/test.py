
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
model = GAN_generator(D_in,H_gen,D_out)
x=torch.randn(D_in)
#generated scene
scene_gen=model(x)
cloudsat_scenes= create_dataset()
scene_real = torch.from_numpy(cloudsat_scenes[110])
scene_real = scene_real.view(1,1,64,64)

H_disc=[256, 128, 128, 5, 6, 64, 128, 256, 256, 4096, 1]
D_in=[1, 5, 1,64]
D_out=[64,64]
N=15
scene=torch.randn([1,1,64,64])
model = GAN_discriminator(D_in,H_disc,D_out)
x=torch.randn(D_in)
#probability that scene_gen is fake
prob=model(x,scene_real)
print(prob)
# script to plot the generated cloudsatimage
# xplot=range(0,64)
# yplot=range(0,64)
# from matplotlib.colors import Normalize
# f,ax = plt.subplots(1, 1)
# norm = Normalize(-1, 1)
# scene_gen_data = scene_gen.data.numpy()
# pcm = ax.pcolormesh(xplot, yplot , (scene_gen_data[0][0]), norm=norm)
#
# plt.show()

# print(len(y_pred[0]))
# print(len(y_pred[0][0]))
# print(len(y_pred[0][0][0]))
