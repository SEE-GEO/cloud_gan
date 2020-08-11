import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy import constants
import h5py
import torch



location = './modis_cloudsat_data/test_data/'
hf = h5py.File(location+'rr_modis_cloudsat_data_2015_0096.h5', 'r') #TRY FILE NUMBER 0190, 0272, 0096, 0060(modis can't handle high clouds)
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
emissivity = np.concatenate([emissivity[:,:,0:3],emissivity[:,:,4:9]],2) #Using eight bands
#emissivity = emissivity[:,:,0:9] #Using all bands except for the last one that contains missing values

central_frequencies = np.array([6.715, 7.325, 8.550, 9.730, 11.030, 12.020, 13.335, 13.635, 13.935, 14.235])*1e-6 #remove the two bands that are not used
central_frequencies = np.concatenate([central_frequencies[0:3], central_frequencies[4:9]]) #Take out only the central frequencies for the bands used
print(central_frequencies)
boltzmann = constants.k
planck = constants.h
speed_of_light = constants.c
print('Modis shape ', emissivity.shape)

modis_temperature = np.zeros([len(emissivity),len(emissivity[0]),len(emissivity[0][0])])


for a in range (0,len(emissivity)):
    for b in range (0,len(emissivity[0])):
        for i in range(0,8):
        #for i in range(0,9):
            #if (2*planck*speed_of_light**2)/((central_frequencies[i]**5)*emissivity[a,b,i])+1 < 1.05:
            if emissivity[a,b,i] == 0:
                print('a,b,i ',a, ' ', b, ' ', i, ' ', emissivity[a,b,i]*1e-6) #'CloudSat value: ', rr[a,b,:]
                #print('a,b,i ',a, ' ', b, ' ', i, ' ', (2*planck*speed_of_light**2)/((central_frequencies[i]**5)*emissivity[a,b,i])+1)
            modis_temperature[a,b,i] = (planck*speed_of_light/(boltzmann*central_frequencies[i]))/np.log((2*planck*speed_of_light**2)/((central_frequencies[i]**5)*emissivity[a,b,i])+1)


altitudes = np.zeros([len(emissivity),64])
for scene in range(0,len(emissivity)):
    for i in range(0,64):
        first_cloud_location = 0
        for j in range(63, -1, -1): #Loop backwards, starting from top of scene
            if rr[scene,i,j] >= -21 and first_cloud_location == 0: #Set dBZ limit for cloud top detection
                altitudes[scene,i] = j #save the index of the altitude where the cloud top is for each position
                first_cloud_location =+1

print('Altitude indices of cloud tops: ', altitudes[0])
altitudes = (altitudes*0.24)+1 #Altitude of cloud top over sea level, [km]
print('Altitudes of cloud tops:', altitudes[0], ' [km]')

#Earth atmosphere model of troposphere
T_0 = 288.15 #Mean temperature at sea level, [K]
temp_from_alt = T_0 - (6.49/1000)*altitudes*1e3
print('Shape modis temp ',modis_temperature.shape)
#print(rr[0,0,:])

alt_from_temp = (T_0-modis_temperature)*(1/6.49)
print('Shape modis altitude ',alt_from_temp.shape)

xplot = np.linspace(0, 64 * 1.1, 64)
#yplot=range(0,8)
yplot=range(0,9)
from matplotlib.colors import Normalize
f1,axs1 = plt.subplots(3,3)
f2,axs2 = plt.subplots(3,3)
#f3,axs3 = plt.subplots(1,1)
f4,axs4 = plt.subplots(3,3)
f5,axs5 = plt.subplots(1,1)
#ax=axs[range(0,3),range(0,3)]

for i in range(0,3):
    for j in range(0,3):
        #pcm=axs1[i,j].pcolormesh(xplot,yplot, np.transpose(emissivity[i*5 + j]))
        pcm = axs1[i, j].pcolormesh(xplot, yplot, np.transpose(modis_temperature[i * 3 + j]))
        title_str = 'Scene' + str(i*3+j + 1)
        axs1[i,j].set_title(title_str, fontsize=4)
        cb = f1.colorbar(pcm, ax=axs1[i,j])
        cb.set_label('emissivity', fontsize=4)
        axs1[i,j].tick_params(axis='both', which='major', labelsize='4')
        cb.ax.tick_params(labelsize=4)


xplot = np.linspace(0, 64 * 1.1, 64)
yplot = np.linspace(1, 16.36, 64)
#ax=axs[range(0,3),range(0,3)]
norm = Normalize(-35, 20)
for i in range(0,3):
    for j in range(0,3):
        pcm=axs2[i,j].pcolormesh(xplot,yplot, np.transpose(rr[i*3 + j]), norm=norm)
        title_str = 'Scene' + str(i*3+j + 1)
        axs2[i,j].set_title(title_str, fontsize=4)
        cb2 = f2.colorbar(pcm, ax=axs2[i,j])
        cb2.set_label('Reflectivites [dBZ]', fontsize=4)
        axs2[i,j].tick_params(axis='both', which='major', labelsize='4')
        axs2[i,j].plot(xplot, altitudes[i*3 + j], color='red', linewidth=0.4)
        for band in range(0, 8):
        #for band in range(0, 9):
            axs2[i,j].plot(xplot, alt_from_temp[i*3 + j, :, band], color='white', linewidth=0.15)
            axs4[i,j].plot(xplot, modis_temperature[i*3 + j, :, band], color='grey', linewidth=1, label='M' + str(band))
        cb2.ax.tick_params(labelsize=4)

        axs4[i,j].plot(xplot, temp_from_alt[i*3+j], color='black', linewidth=1, label='CS')  # Cloud top temperature
        #axs4[i,j].legend()
        title_str4 = 'Scene' + str(i*3+j + 1)
        axs4[i, j].set_title(title_str4, fontsize=4)
        axs4[i,j].set_xlabel("Position [km]", fontsize=4)
        axs4[i, j].tick_params(axis='both', which='major', labelsize='4')
        #axs4[i,j].set_ylabel("Temperature [K]")


'''
pcm=axs3.pcolormesh(xplot,yplot, np.transpose(rr[5]), norm=norm)
title_str = 'Scene' + str(5)
axs3.set_title(title_str, fontsize=10)
cb3 = f3.colorbar(pcm, ax=axs3)
cb3.set_label('Reflectivites [dBZ]', fontsize=10)
axs3.tick_params(axis='both', which='major', labelsize='10')
cb3.ax.tick_params(labelsize=10)
axs3.set_xlabel("Position [km]")
axs3.set_ylabel("Altitude [km]")
'''

pcm=axs5.pcolormesh(xplot,yplot, np.transpose(rr[5]), norm=norm)
title_str = 'Scene' + str(5)
axs5.set_title(title_str, fontsize=10)
cb5 = f5.colorbar(pcm, ax=axs5)
cb5.set_label('Reflectivites [dBZ]', fontsize=10)
axs5.tick_params(axis='both', which='major', labelsize='10')
cb5.ax.tick_params(labelsize=10)
axs5.plot(xplot,altitudes[5], color='red', linewidth=0.7)
axs5.set_xlabel("Position [km]")
axs5.set_ylabel("Altitude [km]")
for i in range(0,8):
#for i in range(0, 9):
    axs5.plot(xplot,alt_from_temp[5,:,i], color='white',linewidth=0.3)


plt.show()
