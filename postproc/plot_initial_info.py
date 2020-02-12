# import plotting
import init
import abc_code.data as data
import logging
import os
import glob
import numpy as np
from numpy.fft import fftfreq, fftn, ifftn

import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
from matplotlib import cm
import numpy as np
import string

# plt.style.use('dark_background')

# # thesis
# single_column = 235
# oneandhalf_column = 352
# double_column = 470

# paper twocolumn elsevair
single_column = 252
oneandhalf_column = 397
double_column = 522
text_height = 682/ 72.27

def fig_size(width_column):
    fig_width_pt = width_column
    inches_per_pt = 1.0 / 72.27  # Convert pt to inches
    golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = fig_width * golden_mean  # height in inches
    return fig_width, fig_height

mpl.rcParams['font.size'] = 9
mpl.rcParams['axes.titlesize'] = 1.2 * plt.rcParams['font.size']
mpl.rcParams['axes.labelsize'] = plt.rcParams['font.size']
mpl.rcParams['legend.fontsize'] = plt.rcParams['font.size']
mpl.rcParams['xtick.labelsize'] = 0.8*plt.rcParams['font.size']
mpl.rcParams['ytick.labelsize'] = 0.8*plt.rcParams['font.size']
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rc('text', usetex=True)
# plt.rcParams['savefig.dpi'] = 2 * plt.rcParams['savefig.dpi']
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.width'] = 0.5
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.width'] = 1
# mpl.rcParams['legend.frameon'] = False
# plt.rcParams['legend.loc'] = 'center left'
plt.rcParams['axes.linewidth'] = 1
#######################################################################################################################
#
#######################################################################################################################
def imagesc(Arrays, map_bounds, titles, name=None):

    Arrays = np.array(Arrays)
    titles = np.array(titles)
    cmap = plt.cm.jet  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]  # extract all colors from the .jet map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)  # create the new map
    norm = mpl.colors.BoundaryNorm(map_bounds, cmap.N)

    axis = [0, 2 * np.pi, 0, 2 * np.pi]
    ticks = ([0, np.pi, 2*np.pi], ['0', 'r$\pi$', 'r$2\pi$'])
    if len(Arrays) > 1:
        fig_width, fig_height = fig_size(single_column)
        print(Arrays.shape)
        nrows, ncols, _, _ = Arrays.shape
        print('nrows, ncols', nrows, ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, figsize=(fig_width, fig_width))
        im = []
        for row in range(nrows):
            for col in range(ncols):
                im. append(axes[row, col].imshow(Arrays[row, col].T, origin='lower', cmap=cmap,
                                                 norm=norm, interpolation="nearest", extent=axis))
                # axes[row, col].set_title(titles[row, col])
                axes[row, col].set_adjustable('box-forced')
                axes[row, col].xaxis.set_ticks(ticks)
                axes[row, col].yaxis.set_ticks(ticks)
                if row == nrows-1:
                    axes[row, col].set_xlabel(r'$x$')
                else:
                    axes[row, col].xaxis.set_major_formatter(plt.NullFormatter())
                if col == 0:
                    axes[row, col].set_ylabel(r'$y$')
                else:
                    axes[row, col].yaxis.set_major_formatter(plt.NullFormatter())
        # cbar_ax = fig.add_axes([0.89, 0.18, 0.017, 0.68])  # ([0.85, 0.15, 0.05, 0.68])
        cbar_sigma = fig.add_axes([0.83, 0.45, 0.017, 0.67])
        cbar_prod = fig.add_axes([0.83, 0.2, 0.017, 0.43])
        fig.subplots_adjust(left=0.11, right=0.8, wspace=0.1, bottom=0.2, top=0.9)
        fig.colorbar(im[0], cax=cbar_sigma, ax=axes.ravel().tolist())
        fig.colorbar(im[-1], cax=cbar_prod, ax=axes.ravel().tolist())
    fig.savefig(name)
    plt.close('all')


########################################################################################################################
# Plot initial data info
########################################################################################################################
def plot_spectra(folder):
    fig = plt.figure(figsize=(4, 3))
    ax = plt.gca()
    files = glob.glob(folder + '*.spectra')

    labels = ['DNS', 'LES', 'test']

    for k in range(len(files)):
        f = open(files[k], 'r')
        data = np.array(f.readlines()).astype(np.float)
        x = np.arange(len(data))
        ax.loglog(x, data, '-', linewidth=2, label=labels[k])

    y = 7.2e14 * np.power(x, -5 / 3)
    ax.loglog(x, y, 'r--', label=r'$-5/3$ slope')
    ax.set_title('Spectra')
    ax.set_ylabel(r'$E$')
    ax.set_xlabel(r'k')
    ax.axis(ymin=1e6)
    plt.legend(loc=0)

    fig.subplots_adjust(left=0.16, right=0.95, bottom=0.2, top=0.87)
    fig.savefig(folder + 'spectra')


def sigma_production_fields(field, dx):
    sigma = deviatoric_stresses(field)
    prod = production(field, dx)
    map_bounds = np.linspace(-0.2, 0.2, 9)

    name = os.path.join(folder, 'sigma_production')
    titles = [[r'$\sigma_{11}$', r'$\sigma_{12}$'], [r'$\sigma_{13}$', r'$\sigma_{ij}\widetilde{S}_{ij}$']]

    imagesc([[sigma['uu'][:, :, 127], sigma['uv'][:, :, 127]], [sigma['uw'][:, :, 127], prod[:, :, 127]]],
            map_bounds, name=name, titles=titles)


def sum_stat(les, test, bins, domain, folder, name):

    labels = ['LES', 'test']

    if name == 'sigma_pdf_log':
        fig, axarr = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(6.5, 2.4))
        if test:
            titles = [r'$\sigma_{11},\ \widehat{\sigma}_{11}$',
                    r'$\sigma_{12},\ \widehat{\sigma}_{12}$',
                    r'$\sigma_{13},\ \widehat{\sigma}_{13}$']
        else:
            titles = [r'$\sigma_{11}$', r'$\sigma_{12}$', r'$\sigma_{13}$']

        for ind, i in enumerate(['uu', 'uv', 'uw']):
            data = les.sum_stat_true[i].flatten()
            x, y = utils.pdf_from_array_with_x(data, bins, domain)
            axarr[ind].plot(x, y, 'r', linewidth=2, label=labels[0])
            axarr[ind].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        if test:
            for ind, i in enumerate(['uu', 'uv', 'uw']):
                data = test.sum_stat_true[i].flatten()
                x, y = utils.pdf_from_array_with_x(data, bins, domain)
                axarr[ind].plot(x, y, 'g', linewidth=2, label=labels[1])
                axarr[ind].set_xlabel(titles[ind])

        axarr[0].axis(xmin=-1.1, xmax=1.1, ymin=1e-5)
        axarr[0].set_ylabel('pdf')
        axarr[0].set_yscale('log')
        plt.legend(loc=0)
        fig.subplots_adjust(left=0.1, right=0.95, wspace=0.1, bottom=0.2, top=0.9)

    elif name == 'production_pdf_log':
        fig = plt.figure(figsize=(4, 3))
        ax = plt.gca()
        if test:
            title = r'$\widehat{\widetilde{P}}'
        else:
            title = r'$\widetilde{P}'

        data = les.sum_stat_true.flatten()
        x, y = utils.pdf_from_array_with_x(data, bins, domain)
        ax.plot(x, y, 'r', linewidth=2, label=labels[0])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        if test:
            data = test.sum_stat_true.flatten()
            x, y = utils.pdf_from_array_with_x(data, bins, domain)
            ax.plot(x, y, 'g', linewidth=2, label=labels[1])
            ax.set_xlabel(title)

        ax.axis(xmin=-5, xmax=5, ymin=1e-5)
        ax.set_ylabel('pdf')
        ax.set_yscale('log')
        plt.legend(loc=0)
        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)
    fig.savefig(os.path.join(folder, name))
    plt.close('all')

#######################################################################################################################
#
#######################################################################################################################
logging.basicConfig(format='%(levelname)s: %(name)s: %(message)s', level=logging.DEBUG)

spectra = 0

data_name = 'JHU_data'

folder = os.path.join('../ABC/plots/data_plots/JHU_data/')
folder_data = os.path.join('../ABC/data_input/JHU_data/')
if not os.path.isdir(folder):
    os.makedirs(folder)
    load = 0
if not os.path.isdir(folder_data):
    logging.error('Incorrect data path: {}'.format(folder_data))
    exit()
logging.debug('\ninput: {}\noutput: {}'.format(folder_data, folder))


def shell_average(spect3D, N_point, k_3d):
    """ Compute the 1D, shell-averaged, spectrum of the 3D Fourier-space.
    :param spect3D: 3-dimensional complex or real Fourier-space scalar
    :param k_3d:  wavemode of each 3D wavevector
    :return: 1D, shell-averaged, spectrum
    """
    i = 0
    F_k = np.zeros(N_point**3)
    k_array = np.empty_like(F_k)
    for ind_x, kx in enumerate(k_3d[0]):
        for ind_y, ky in enumerate(k_3d[1]):
            for ind_z, kz in enumerate(k_3d[2]):
                k_array[i] = round(np.sqrt(kx**2 + ky**2 + kz**2))
                F_k[i] = 2*np.pi*k_array[i]**2*spect3D[ind_x, ind_y, ind_z]
                i += 1
    all_F_k = sorted(list(zip(k_array, F_k)))

    x, y = [all_F_k[0][0]], [all_F_k[0][1]]
    n = 1
    for k, F in all_F_k[1:]:
        if k == x[-1]:
            n += 1
            y[-1] += F
        else:
            y[-1] /= n
            x.append(k)
            y.append(F)
            n = 1
    return x, y


def spectral_density(vel_array, dx, N_points, fname):
    """
    Write the 1D power spectral density of var to text file. Method
    assumes a real input in physical space.
    """
    k = 2*np.pi*np.array([fftfreq(N_points[0], dx[0]), fftfreq(N_points[1], dx[1]), fftfreq(N_points[2], dx[2])])
    spect3d = 0
    for array in vel_array:
        fft_array = fftn(array)
        spect3d += np.real(fft_array * np.conj(fft_array))

    x, y = shell_average(spect3d, N_points[0], k)
    fh = open(fname + '.spectra', 'w')
    fh.writelines(["%s\n" % item for item in y])
    fh.close()


def load_HIT_data(params):
    datafile = dict()
    if data_name == 'JHU_data':
        datafile['u'] = os.path.join(folder_data, 'HIT_u.bin')
        datafile['v'] = os.path.join(folder_data, 'HIT_v.bin')
        datafile['w'] = os.path.join(folder_data, 'HIT_w.bin')
        type_of_bin_data = np.float32
    elif data_name == 'CU_data':
        datafile['u'] = os.path.join(folder_data, 'Velocity1_003.rst')
        datafile['v'] = os.path.join(folder_data, 'Velocity2_003.rst')
        datafile['w'] = os.path.join(folder_data, 'Velocity3_003.rst')
        type_of_bin_data = np.float64

    HIT_data = dict()
    data_shape = (params.physical_case['N_point'], params.physical_case['N_point'], params.physical_case['N_point'])
    for i in ['u', 'v', 'w']:
        HIT_data[i] = np.reshape(np.fromfile(datafile[i], dtype=type_of_bin_data), data_shape)
    for key, value in HIT_data.items():
        HIT_data[key] = np.swapaxes(value, 0, 2)  # to put x index in first place
    for i in ['u', 'v', 'w']:
        for j in ['u', 'v', 'w']:
            HIT_data[i + j] = np.multiply(HIT_data[i], HIT_data[j])

    dx = [params.physical_case['lx'] / params.physical_case['N_point']] * 3
    if spectra:
        logging.info('calculate DNS spectra')
        spectral_density([HIT_data['u'], HIT_data['v'], HIT_data['w']], dx, params.physical_case['N_point'], folder + 'DNS')
    return HIT_data


def filter3d(data, scale_k, dx, N_points):
    """ Tophat filter in Fourier space for dictionary of 3D arrays.
        data - dictionary of numpy arrays;
        scale_k - wave number, which define size of filter."""

    def tophat_kernel(k, limit):
        """Create 3D array of Tophat filter.
            k - array of wave numbers;
            limit - cutoff wavenumber."""
        a = np.zeros((len(k[0]), len(k[1]), len(k[2])), dtype=np.float32)
        for indx, kx in enumerate(k[0]):
            for indy, ky in enumerate(k[1]):
                for indz, kz in enumerate(k[2]):
                    a[indx, indy, indz] = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)

        kernel = np.piecewise(a, [a <= limit, a > limit], [1, 0])
        return kernel

    FFT = dict()
    for key, value in data.items():
        FFT[key] = fftn(value)
    k = [fftfreq(N_points[0], dx[0]), fftfreq(N_points[1], dx[1]), fftfreq(N_points[2], dx[2])]
    # Filtering
    kernel = tophat_kernel(k, scale_k)

    result = dict()
    fft_filtered = dict()
    for key, value in FFT.items():
        fft_filtered[key] = np.multiply(value, kernel)

    for key, value in fft_filtered.items():
        result[key] = ifftn(value).real

    return result

def field_gradient(field, dx):
    """Calculate tensor of gradients of self.field.
    :return:      dictionary of gradient tensor
    """
    grad = dict()
    grad['uu'], grad['uv'], grad['uw'] = np.gradient(field['u'], dx[0], dx[1], dx[2])
    grad['vu'], grad['vv'], grad['vw'] = np.gradient(field['v'], dx[0], dx[1], dx[2])
    grad['wu'], grad['wv'], grad['ww'] = np.gradient(field['w'], dx[0], dx[1], dx[2])
    return grad

def calc_strain_tensor(field, dx):
    """Calculate strain tensor S_ij = 1/2(du_i/dx_j+du_j/dx_i) of given field.
    :return:      dictionary of strain tensor
    """
    A = field_gradient(field, dx)
    tensor = dict()
    for i in ['u', 'v', 'w']:
        for j in ['u', 'v', 'w']:
            tensor[i + j] = 0.5 * (A[i + j] + A[j + i])
    return tensor


def deviatoric_stresses(field):
    """Calculate pdf of deviatoric stresses using DNS data.
        sigma_ij = tau_ij - 1/3 tau_kk*delta_ij
    :return:     dictionary of deviatoric stresses
    """
    tau = dict()
    for i in ['u', 'v', 'w']:
        for j in ['u', 'v', 'w']:
            tau[i + j] = field[i + j] - np.multiply(field[i], field[j])
    trace = tau['uu'] + tau['vv'] + tau['ww']
    for i in ['uu', 'vv', 'ww']:
        tau[i] -= 1 / 3 * trace
    return tau

def production(field, dx):
    sigma = deviatoric_stresses(field)
    prod_rate = 0
    S = calc_strain_tensor(field, dx)
    for i in ['u', 'v', 'w']:
        for j in ['u', 'v', 'w']:
            prod_rate += sigma[i + j] * S[i + j]
    return prod_rate
#######################################################################################################################
#
#######################################################################################################################


def main():


    params = init.CreateParams(path=os.path.join('../', 'params.yml'))
    print(params)
    dx = [params.physical_case['lx']/params.physical_case['N_point']] * 3
    logging.info('Load HIT data')
    HIT_data = load_HIT_data(params)


    logging.info('Filter HIT data')
    LES_data = filter3d(data=HIT_data, scale_k=params.physical_case['LES_scale'], dx=dx,
                        N_points=[params.physical_case['N_point']]*3)
    TEST_data = None
    if params.physical_case['TEST_scale']:
        TEST_data = filter3d(data=HIT_data, scale_k=params.physical_case['TEST_scale'], dx=dx,
                             N_points=[params.physical_case['N_point']]*3)

    LES_delta = 1 / params.physical_case['LES_scale']
    # LES = data.Data(LES_data, LES_delta, dx, params.compare_pdf)
    # TEST = None
    if params.physical_case['TEST_scale']:
        TEST_delta = 1 / params.physical_case['TEST_scale']
    #     TEST = data.Data(TEST_data, TEST_delta, dx, params.compare_pdf)

    if spectra:
        logging.info('Calculate LES spectra')
        spectral_density([LES_data['u'], LES_data['v'], LES_data['w']], dx,
                         params.physical_case['N_point'], folder + 'LES')
        if params.physical_case['TEST_scale']:
            logging.info('Calculate TEST spectra')
            spectral_density([TEST_data['u'], TEST_data['v'], TEST_data['w']], dx,
                             params.physical_case['N_point'], folder + 'test')

    # logging.info('Plot initial data info')
    # if spectra:
    #     plot_spectra(folder)
    # map_bounds = np.linspace(np.min(LES_data['v'][:, :, 127]), np.max(LES_data['v'][:, :, 127]), 9)
    # plotting.compare_filter_fields(HIT_data, LES_data, TEST_data, map_bounds, folder=folder)
    # plotting.vel_fields(HIT_data, scale='DNS', map_bounds=map_bounds, folder=folder)
    # plotting.vel_fields(LES_data, scale='LES', map_bounds=map_bounds, folder=folder)
    # if params.physical_case['TEST_scale']:
    #     plotting.vel_fields(TEST_data, scale='TEST', map_bounds=map_bounds, folder=folder)
    dx = np.array([params.physical_case['lx'] / params.physical_case['N_point']] * 3)
    sigma_production_fields(LES_data, dx)


if __name__ == '__main__':
    main()



########################################################################################################################
# Initial data
########################################################################################################################
# fig = pickle.load(open(params.plot_folder + 'LES_velocities', 'rb'))
# plt.show()
#
# fig = pickle.load(open(params.plot_folder + 'TEST_velocities', 'rb'))
# plt.show()
#
# fig = pickle.load(open(params.plot_folder+'sigma_TEST', 'rb'))
# fig.set_title[r'$\sigma_{11}$', r'$\sigma_{12}$', r'$\sigma_{13}$']
# plt.show()
#
# fig = pickle.load(open(params.plot_folder+'TEST', 'rb'))
#
# plt.show()