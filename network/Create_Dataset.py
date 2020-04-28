import os
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
from datetime import datetime

from wxdata import Index

dendrite_path = "~/Dendrite"
index = Index.load(os.path.join(dendrite_path,
                                "SatData/CloudSat/2b_geoprof.index"))

start = datetime(2015, 1, 1)
end   = datetime(2016, 1, 1)

files = index.get_files("CloudSat_2b_GeoProf",
                        start = start,
                        end = end)
ind=0;
file = files[ind].open()

z = file.altitude
x = file.latitude
x = np.broadcast_to(x.reshape(-1, 1), z.shape)
rr = file.radar_reflectivity


rr_data_element=np.ones([(int)(len(z)/64),64,64])*(-40)
mask = rr_data_element == -8888
rr_data_element =ma.masked_array(rr_data_element,mask=mask)
# should we change the mask?
for i in range(0, len(z)-64,64):

    for k in range(0,64):
        j = 124
        while z[i+k][j]<500:
            j=j-1

        n=j-64
        test=(int) (i/64)
        rr_data_element[test][k]=rr[i+k][j:n:-1]

            #print(rr[i+k][l])
print(type(rr_data_element))
print(type(rr))
xplot=range(0,64)
yplot=range(0,64)
from matplotlib.colors import Normalize
f,ax = plt.subplots(1, 1)
norm = Normalize(-30, 30)
pcm = ax.pcolormesh(xplot, yplot , np.transpose(rr_data_element[250]), norm=norm)

plt.show()