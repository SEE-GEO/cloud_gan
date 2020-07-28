import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy import constants
import h5py

location = './modis_cloudsat_data/test_data/'
hf = h5py.File(location+'rr_modis_cloudsat_data_2015_0190.h5', 'r')
rr=hf.get('rr')
rr = np.array(rr)
emissivity = hf.get('emissivity')
emissivity = np.array(emissivity)
emissivity_mask = emissivity == -40
emissivity= ma.masked_array(emissivity,mask=emissivity_mask)
start_time = hf.get('start')
start_time = np.array(start_time)
print(start_time)
print(emissivity.shape)
print(rr.shape)
rr=(rr +1)*(55/2)-35
emissivity = emissivity*1e6
central_frequencies = np.array([6.715, 7.325, 8.550, 9.730, 11.030, 12.020, 13.335, 13.635, 13.935, 14.235])*1e-6
boltzmann = constants.k
planck = constants.h
speed_of_light = constants.c
print('Modis shape ', emissivity.shape)

modis_temperature = np.zeros([len(emissivity),len(emissivity[0]),len(emissivity[0][0])])

print('testvalue 1 ', emissivity[0, 5, 9])
print('testvalue 2 ', emissivity[0, 14, 9])
print(len(emissivity))
for a in range (0,len(emissivity)):
    for b in range (0,len(emissivity[0])):
        for i in range(0,10):
            #if (2*planck*speed_of_light**2)/((central_frequencies[i]**5)*emissivity[a,b,i])+1 < 1.05:
            if emissivity[a,b,i] == 0:
                print('a,b,i ',a, ' ', b, ' ', i, ' ', emissivity[a,b,i]*1e-6) #'CloudSat value: ', rr[a,b,:]
                #print('a,b,i ',a, ' ', b, ' ', i, ' ', (2*planck*speed_of_light**2)/((central_frequencies[i]**5)*emissivity[a,b,i])+1)
            modis_temperature[a,b,i] = (planck*speed_of_light/(boltzmann*central_frequencies[i]))/np.log((2*planck*speed_of_light**2)/((central_frequencies[i]**5)*emissivity[a,b,i])+1)


xplot=range(0,64)
yplot=range(0,10)
from matplotlib.colors import Normalize
f1,axs1 = plt.subplots(5,5)
#ax=axs[range(0,3),range(0,3)]

for i in range(0,5):
    for j in range(0,5):
        #pcm=axs1[i,j].pcolormesh(xplot,yplot, np.transpose(emissivity[i*5 + j]))
        pcm = axs1[i, j].pcolormesh(xplot, yplot, np.transpose(modis_temperature[i * 5 + j]))
        title_str = 'Scene' + str(i*5+j + 1)
        axs1[i,j].set_title(title_str, fontsize=2)
        cb = f1.colorbar(pcm, ax=axs1[i,j])
        cb.set_label('emissivity', fontsize=2)
        axs1[i,j].tick_params(axis='both', which='major', labelsize='2')
        cb.ax.tick_params(labelsize=2)

xplot=range(0,64)
yplot=range(0,2)
ftest,axtest = plt.subplots(3,3)
for i in range(0,3):
    for j in range(0,3):

        test = axtest[i,j].pcolormesh(xplot, yplot, np.transpose(modis_temperature[i * 5 + j,:,3:5]))
        title_str = 'Scene' + str(i*5+j + 1)
        axtest[i,j].set_title(title_str, fontsize=4)
        axtest[i,j].set_ylabel("Channel")
        axtest[i,j].set_xlabel("Position")
        cb_test = ftest.colorbar(test, ax=axtest[i,j])
        cb_test.set_label('Temperature [K]', fontsize=4)


xplot=range(0,64)
yplot=range(0,64)

from matplotlib.colors import Normalize
f2,axs2 = plt.subplots(5,5)
#ax=axs[range(0,3),range(0,3)]
norm = Normalize(-35, 20)
for i in range(0,5):
    for j in range(0,5):
        pcm=axs2[i,j].pcolormesh(xplot,yplot, np.transpose(rr[i*5 + j]), norm=norm)
        title_str = 'Scene' + str(i*5+j + 1)
        axs2[i,j].set_title(title_str, fontsize=2)
        cb2 = f2.colorbar(pcm, ax=axs2[i,j])
        cb2.set_label('Reflectivites [dBZ]', fontsize=2)
        axs2[i,j].tick_params(axis='both', which='major', labelsize='2')
        cb2.ax.tick_params(labelsize=2)
plt.show()

