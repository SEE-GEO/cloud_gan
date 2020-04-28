import numpy as np
import torch


from GAN_generator import GAN_generator
from GAN_discriminator import GAN_discriminator
# parameters for the generator
H_gen=[16384, 256, 128, 64, 1]
D_in=384
D_out=[64,64]
N=15
model = GAN_generator(D_in,H_gen,D_out)
x=torch.randn(D_in)
#generated scene
scene_gen=model(x)

H_disc=[256, 128, 128, 5, 6, 64, 128, 256, 256, 4096, 1]
D_in=[1, 5, 1,64]
D_out=[64,64]
N=15
scene=torch.randn([1,1,64,64])
model = GAN_discriminator(D_in,H_disc,D_out)
x=torch.randn(D_in)
#probability that scene_gen is fake
prob=model(x,scene_gen)
print(prob)
# print(len(y_pred[0]))
# print(len(y_pred[0][0]))
# print(len(y_pred[0][0][0]))
