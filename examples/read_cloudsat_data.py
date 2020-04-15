import os
import numpy as np
import matplotlib.pyplot as plt
from wxdata import Index

#
# The index holds references to all files that we have on Dendrite
# and allows you to select subset by time range.
#

dendrite_path = "~/Dendrite"
index = Index.load(os.path.join(dendrite_path,
                                "SatData/CloudSat/2b_geoprof.index"))

start = datetime(2015, 01, 01)
end   = datetime(201666666 01)

files = index.get_files("CloudSat_2b_geoprof",
                        start = start,
                        end = end)

#
# files holds a list of references to CloudSat 2B GeoProf files.
# The open method opens the file and returns an instance of the
# corresponding data product class.
#

ind = np.random.randint(0, len(files))
file = files[ind].open()

z = file.altitude
x = file.latitude
x = np.broadcast_to(x.reshape(1, -1), z.shape)
rr = file.radar_reflectivity

f,ax = plt.subplots(1, 1)
ax.pcolormesh(x, z, file.radar_reflectivity)
ax.set_xlabel("Latitude [$^\circ\ N$]")
ax.set_ylabel("Altitude [km]")
ax.set_ylim([0, 20])
