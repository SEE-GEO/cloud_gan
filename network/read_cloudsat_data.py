import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from wxdata import Index

#
# The index holds references to all files that we have on Dendrite
# and allows you to select subset by time range.
#

dendrite_path = "~/Dendrite"
index = Index.load(os.path.join(dendrite_path,
                                "SatData/CloudSat/2b_geoprof.index"))

start = datetime(2015, 1, 1)
end   = datetime(2016, 1, 1)

files = index.get_files("CloudSat_2b_GeoProf",
                        start = start,
                        end = end)

#
# files holds a list of references to CloudSat 2B GeoProf files.
# The open method opens the file and returns an instance of the
# corresponding data product class.
#

ind = 0
file = files[ind].open()

#
# Radar reflectivity, altitude, latitude and longitued armetop2034CSe exposed as a direct attributes
# of the product class.
#

z = file.altitude
x = file.latitude
x = np.broadcast_to(x.reshape(-1, 1), z.shape)
qa = file.radar_reflectivity
cloud_mask=file.cloud_mask
print(qa[0])
print(len(z))
print(len(x))
print(len(qa[0]))
from matplotlib.colors import Normalize
f,ax = plt.subplots(1, 1)
norm = Normalize(-30, 30)
pcm = ax.pcolormesh(x, z / 1e3, file.radar_reflectivity[150], norm=norm)
ax.set_xlabel("Latitude [$^\circ\ N$]")
ax.set_xlim([-90, 90])
ax.set_ylabel("Altitude [km]")
ax.set_ylim([0, 20])
#im=ax.imshow(rr,cmap='gist_earth')
cb=f.colorbar(pcm,ax=ax)
cb.set_label('Reflectivities [dBZ]')
plt.show()
#
# The remaining data attributes can be accessed using file[<attribute_name>]. Use
# file.attributes to list the available entries.
#

print(file.attributes)
