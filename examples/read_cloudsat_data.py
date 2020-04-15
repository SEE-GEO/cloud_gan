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

ind = np.random.randint(0, len(files))
file = files[ind].open()

#
# Radar reflectivity, altitude, latitude and longitued are exposed as a direct attributes
# of the product class.
#

z = file.altitude
x = file.latitude
x = np.broadcast_to(x.reshape(-1, 1), z.shape)
rr = file.radar_reflectivity

from matplotlib.colors import Normalize
f,ax = plt.subplots(1, 1)
norm = Normalize(-30, 30)
ax.pcolormesh(x, z / 1e3, file.radar_reflectivity, norm=norm)
ax.set_xlabel("Latitude [$^\circ\ N$]")
ax.set_xlim([-30, 30])
ax.set_ylabel("Altitude [km]")
ax.set_ylim([0, 20])

plt.show()

#
# The remaining data attributes can be accessed using file[<attribute_name>]. Use
# file.attributes to list the available entries.
#

print(file.attributes)
