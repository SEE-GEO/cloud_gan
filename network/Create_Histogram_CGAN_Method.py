import datetime

import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from os import path
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# Create one file with all cloudsat_scenes
from GAN_generator import GAN_generator
import matplotlib.ticker as ticker


def Create_Histogram_CGAN_Method(modis_uncertainty,number):
    # modis_uncertainty == None, number = 0 if no uncertainty
    # number = 1 for minimum uncertainty
    # number = 3 for maximum uncertainty
    n_bins = 4 * 55
    n_removed = int((30 / 165) * n_bins)

    #location = './modis_cloudsat_data/2016/' #for CGAN 2016
    #file_string = location + 'modis_cloudsat_ElevLatLong_test_data_conc_2016_normed2015_ver2.h5' #for CGAN 2016

    file_string = './CGAN_test_data_with_temp_conc_ver2' + '.h5' #for CGAN 2015
    hf = h5py.File(file_string, 'r')

    temperatures = torch.tensor(hf.get('temperature'))

    cloudsat_scenes = torch.tensor(hf.get('cloudsat_scenes'))
    cloudsat_scenes = (cloudsat_scenes + 1) * (55 / 2) - 35
    cloudsat_scenes = cloudsat_scenes.view(-1, 64)
    modis_scenes = torch.tensor(hf.get('modis_scenes'))
    print('shape of modis scenes: ', modis_scenes.shape)

    print(len(cloudsat_scenes), " files loaded: ")
    temp_modis_scenes = torch.cat([modis_scenes[:, :, :, 0:3], modis_scenes[:, :, :, 4:9]], 3)
    modis_scenes = temp_modis_scenes

    print('Shape of cloudsat_scenes', cloudsat_scenes.shape)

    total_histogram = np.ones([64, n_bins])
    dbz_per_bin = 55 / n_bins
    for i in range(0, 64):
        scenes_histogram, bin_edges = np.histogram(cloudsat_scenes[:, i], bins=n_bins, range=(-35, 20), density=False)
        total_histogram[i] = scenes_histogram
        total_sum = np.sum(total_histogram[i])

        total_histogram[i] = total_histogram[i] / (dbz_per_bin * total_sum)

    print(total_histogram.shape)

    # total_sum_real = np.sum(total_histogram)

    # total_histogram = total_histogram/(dbz_per_bin*total_sum_real)

    # print(bin_edges)
    new_total = total_histogram[:, n_removed:n_bins]
    print(new_total.shape)

    x = np.linspace(bin_edges[n_removed], 20, n_bins - n_removed)
    y = np.linspace(1, 16.36, 64)
    f, ax = plt.subplots(1, 1, figsize=(10.5, 15))

    pcm = ax.pcolormesh(x, y, new_total, vmin=0, vmax=0.035)
    ax.set_ylabel("Altitude [km]", fontsize=28)
    ax.set_xlabel("Reflectivity [dBZ]", fontsize=28)
    # ax.set_aspect((n_bins - n_removed) / 64)
    ax.set_aspect(155 / 64)
    #ax.set_title('CGAN', fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=26)
    #ax.set_title('CGAN 2016', fontsize=32)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('bottom', size='100%', pad=0.05)
    # cax = f.add_axes([0.09, 0.06, 0.84, 0.02])
    cb = f.colorbar(pcm, ax=ax, orientation='horizontal', fraction=0.049, pad=0.15)
    cb.set_label('Norm. occurrence (Real) [dBZ$^{-1}$]', fontsize=28)
    cb.ax.tick_params(labelsize=26, rotation=45)
    plt.savefig('./Results/ErrorEstimation/histogram_real_CGAN_with_uncertainty'+str(number)+'.png')


    # Load this (below) if you want to create new generated scenes
    #region
    '''
    if number == 0:
        print('No uncertainty used for MODIS')
    else:
        for band in range(0, 8):
            uncertainty = torch.randn(size=[len(modis_scenes), 1, 64]) * float(modis_uncertainty[band] / 100)
            modis_scenes[:, :, :, band] = modis_scenes[:, :, :, band] + torch.mul(modis_scenes[:, :, :, band], uncertainty)

    print('Shape of modis scenes with uncertainty: ', modis_scenes.shape)
    D_gen = [len(cloudsat_scenes)//(64*100),1, 1, 64]
    H_gen = [576, 16384, 256, 128, 64, 1]
    netG = GAN_generator(H_gen)
    folder = './'
    modis_scenes = torch.transpose(modis_scenes, 1, 3)
    modis_scenes = torch.transpose(modis_scenes, 2, 3)
    print(99*len(cloudsat_scenes)//(64*100))
    print(99*len(cloudsat_scenes)//(64*100)+len(cloudsat_scenes)//(64*100))
    print('så här ser modis ut: ',modis_scenes[100*len(cloudsat_scenes)//(64*100):100*len(cloudsat_scenes)//(64*100) + len(cloudsat_scenes)%(64*100),:,:,:].shape)
    if path.exists(folder + 'network_parameters_CGAN_3500.pt'):
        with torch.no_grad():
            checkpoint = torch.load(folder + 'network_parameters_CGAN_3500.pt', map_location=torch.device('cpu'))
            netG.load_state_dict(checkpoint['model_state_dict_gen'])
            netG.zero_grad()
            print('dadada')
            all_generated = []

            for i in range(0,100):
                index1 = i*(len(cloudsat_scenes)//(64*100))
                index2 = (i+1)*(len(cloudsat_scenes)//(64*100))
                print('Test shape ',(modis_scenes[index1:index2, :,:, :]).shape)
                print((torch.randn(D_gen)).shape)
                generated_scenes = netG(torch.randn(D_gen), modis_scenes[index1:index2, :,:, :])
                print(generated_scenes.shape)
                generated_scenes = (torch.transpose(generated_scenes,2,3))
                print(generated_scenes.shape)
                generated_scenes = generated_scenes.reshape(-1, 64)
                generated_scenes = generated_scenes.detach().numpy()
                print(generated_scenes.shape)
                generated_scenes = np.array(generated_scenes)
                print(i)
                print(index1)
                print(index2)
                if i == 0 :
                    all_generated = generated_scenes
                else:
                    all_generated = np.append(all_generated,generated_scenes,axis=0)
            print((len(cloudsat_scenes)%(64*100))//64)
            D_gen = [(len(cloudsat_scenes)%(64*100))//64, 1,1,64]
            print('d_gen size ', len(D_gen))
            #index2=14600
            generated_scenes = netG(torch.randn(D_gen), modis_scenes[index2:, :,:, :])
            generated_scenes = (torch.transpose(generated_scenes, 2, 3))
            generated_scenes = generated_scenes.reshape(-1, 64)
            generated_scenes = generated_scenes.detach().numpy()
            generated_scenes = np.array(generated_scenes)
            print(generated_scenes.shape)
            all_generated = np.append(all_generated,generated_scenes,axis=0)
            all_generated = (all_generated + 1) * (55 / 2) - 35
            print(all_generated.shape)
            #name_string = 'generated_scenes_for_histogram_test_temp_' #for 2015
            #name_string = 'generated_scenes_for_histogram_2016'
            name_string = 'generated_scenes_for_histogram_with_uncertainty_'+str(number)
            filename = 'CGAN_' + name_string + '.h5'
            print(len(all_generated))
            print(len(modis_scenes))
            print(len(temperatures))
            hf = h5py.File(filename, 'w')
            hf.create_dataset('modis_scenes',data=modis_scenes[: len(all_generated)])
            hf.create_dataset('all_generated', data=all_generated)
            hf.create_dataset('temperatures',data = temperatures[: len(all_generated)])
            hf.close()
    '''
    #endregion

    # Load generated scenes in region below
    #region

    print('Starting to load generated scenes')
    location = './'
    file_string = location + 'CGAN_generated_scenes_for_histogram_with_uncertainty_'+str(number)+'.h5'
    hf = h5py.File(file_string, 'r')
    
    all_generated = torch.tensor(hf.get('all_generated'))
    print('Generated scenes with uncertainty loaded')
    #endregion

    hf = h5py.File('CGAN_generated_scenes_for_histogram_test_temp_.h5', 'r')
    all_generated_zero_uncert = torch.tensor(hf.get('all_generated'))
    print('Generated scenes zero uncertainty loaded')


    total_histogram_generated = np.ones([64, n_bins])
    total_histogram_generated_zero = np.ones([64, n_bins])

    for i in range(0, 64):
        scenes_histogram, bin_edges = np.histogram(all_generated[:, i], bins=n_bins, range=(-35, 20), density=False)
        total_histogram_generated[i] = scenes_histogram
        total_sum_generated = np.sum(total_histogram_generated[i])
        total_histogram_generated[i] = total_histogram_generated[i] / (dbz_per_bin * total_sum_generated)

        scenes_histogram_zero, bin_edges = np.histogram(all_generated_zero_uncert[:, i], bins=n_bins, range=(-35, 20), density=False)
        total_histogram_generated_zero[i] = scenes_histogram_zero
        total_sum_generated_zero = np.sum(total_histogram_generated_zero[i])
        total_histogram_generated_zero[i] = total_histogram_generated_zero[i] / (dbz_per_bin * total_sum_generated_zero)

    new_total_generated = total_histogram_generated[:, n_removed:n_bins]
    new_total_generated_zero = total_histogram_generated_zero[:, n_removed:n_bins]

    x = np.linspace(bin_edges[n_removed], 20, n_bins - n_removed)
    y = np.linspace(1, 16.36, 64)
    f, ax = plt.subplots(1, 1, figsize=(10.5, 15))

    pcm = ax.pcolormesh(x, y, new_total_generated, vmin=0, vmax=0.035)
    #ax.set_ylabel("Altitude [km]")
    ax.set_xlabel("Reflectivity [dBZ]", fontsize=28)
    # ax.set_aspect((n_bins - n_removed) / 64)
    if number == 3:
        ax.set_title('Maximum uncertainty', fontsize=32)
    elif number == 2:
        ax.set_title('Average uncertainty', fontsize=32)
    elif number == 1:
        ax.set_title('Minimum uncertainty', fontsize=32)
        ax.set_ylabel("Altitude [km]", fontsize=28)
    ax.set_aspect(155 / 64)
    cb = f.colorbar(pcm, ax=ax, orientation='horizontal', fraction=0.049, pad=0.15)
    cb.set_label('Norm. occurrence (CGAN) [dBZ$^{-1}$]', fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=26)
    cb.ax.tick_params(labelsize=26, rotation=45)
    plt.savefig('./Results/ErrorEstimation/histogram_generated_CGAN_with_uncertainty_'+str(number)+'.png')

    difference_histogram = new_total_generated - new_total
    x = np.linspace(bin_edges[n_removed], 20, n_bins - n_removed)
    y = np.linspace(1, 16.36, 64)
    f, ax = plt.subplots(1, 1, figsize=(10.5, 15))
    pcm = ax.pcolormesh(x, y, difference_histogram, cmap='seismic', vmin=-0.017, vmax=0.017)
    # ax3.set_ylabel("Altitude", fontsize=14)
    if number == 1:
        ax.set_ylabel("Altitude [km]", fontsize=28)
    ax.set_xlabel("Reflectivity [dBZ]", fontsize=28)
    # ax.set_aspect((num_bins-num_removed_bins)/64)
    ax.set_aspect(155 / 64)
    # ax3.set_aspect('auto')
    # ax3.set_title('Occurence difference')
    cb = f.colorbar(pcm, ax=ax, orientation='horizontal', fraction=0.049, pad=0.15)
    cb.set_label('Occurrence bias (CGAN - Real) [dBZ$^{-1}$]', fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=26)
    # cb3.set_ticks(np.arange(-0.0075, 0.01, step=0.0025))
    cb.ax.tick_params(labelsize=26, rotation=45)
    # f3.tight_layout()
    plt.savefig('./Results/ErrorEstimation/Difference_histogram_CGAN_real_and_generated_with_uncertainty_'+str(number)+'.png')

    # Plot difference between generated histogram with and without uncertainty
    uncertainty_difference = new_total_generated_zero - new_total_generated
    x = np.linspace(bin_edges[n_removed], 20, n_bins - n_removed)
    y = np.linspace(1, 16.36, 64)
    f, ax = plt.subplots(1, 1, figsize=(10.5, 15))
    pcm = ax.pcolormesh(x, y, uncertainty_difference, cmap='seismic', vmin=-0.017, vmax=0.017)
    #pcm = ax.pcolormesh(x, y, uncertainty_difference, cmap='seismic', vmin=-0.005, vmax=0.005)
    # ax3.set_ylabel("Altitude", fontsize=14)
    if number == 1:
        ax.set_ylabel("Altitude [km]", fontsize=28)
    ax.set_xlabel("Reflectivity [dBZ]", fontsize=28)
    # ax.set_aspect((num_bins-num_removed_bins)/64)
    ax.set_aspect(155 / 64)
    # ax3.set_aspect('auto')
    # ax3.set_title('Occurence difference')
    cb = f.colorbar(pcm, ax=ax, orientation='horizontal', fraction=0.049, pad=0.15)
    cb.set_label('Occurrence bias (CGAN with error - CGAN) [dBZ$^{-1}$]', fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=26)
    # cb3.set_ticks(np.arange(-0.0075, 0.01, step=0.0025))
    cb.ax.tick_params(labelsize=26, rotation=45)
    # f3.tight_layout()
    plt.savefig('./Results/ErrorEstimation/Difference_histogram_CGAN_and_generated_with_error_' + str(number) + '.png')
    #plt.savefig('./Results/ErrorEstimation/Difference_histogram_CGAN_and_generated_with_error_' + str(number) + '_smaller_scale.png')



    print('done')