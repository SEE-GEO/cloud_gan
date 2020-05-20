
import numpy as np
import matplotlib.pyplot as plt
def plot_cloud(rr):

    xplot=range(0,64)
    yplot=range(0,64)
    from matplotlib.colors import Normalize
    f,ax = plt.subplots(1, 1)
    rr_plot=(rr +1)*(55/2)-35
    norm = Normalize(-35, 20)
    pcm = ax.pcolormesh(xplot, yplot , np.transpose(rr_plot),norm=norm)
    cb=f.colorbar(pcm,ax=ax)
    cb.set_label('Reflectivities [dBZ]')
    plt.show()