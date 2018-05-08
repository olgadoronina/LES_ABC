import params
import plotting
import data
import logging
import os
import numpy as np
from numpy.fft import fftfreq, fftn, ifftn


logging.basicConfig(format='%(levelname)s: %(name)s: %(message)s', level=logging.DEBUG)
spectra = 0
folder = './plots/init_info/'
if not os.path.isdir(folder):
    os.makedirs(folder)


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


def load_HIT_data():
    datafile = dict()
    if params.DATA == 'JHU_data':
        datafile['u'] = params.data_folder + 'HIT_u.bin'
        datafile['v'] = params.data_folder + 'HIT_v.bin'
        datafile['w'] = params.data_folder + 'HIT_w.bin'
        type_of_bin_data = np.float32
    elif params.DATA == 'CU_data':
        datafile['u'] = params.data_folder + 'Velocity1_003.rst'
        datafile['v'] = params.data_folder + 'Velocity2_003.rst'
        datafile['w'] = params.data_folder + 'Velocity3_003.rst'
        type_of_bin_data = np.float64

    HIT_data = dict()
    data_shape = (params.N_point, params.N_point, params.N_point)
    for i in ['u', 'v', 'w']:
        HIT_data[i] = np.reshape(np.fromfile(datafile[i], dtype=type_of_bin_data), data_shape)
    for key, value in HIT_data.items():
        HIT_data[key] = np.swapaxes(value, 0, 2)  # to put x index in first place
    for i in ['u', 'v', 'w']:
        for j in ['u', 'v', 'w']:
            HIT_data[i + j] = np.multiply(HIT_data[i], HIT_data[j])

    dx = np.divide(params.lx, params.N_points)
    if spectra:
        logging.info('calculate DNS spectra')
        spectral_density([HIT_data['u'], HIT_data['v'], HIT_data['w']], dx, params.N_points, folder + 'DNS')
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


def main():

    dx = np.divide(params.lx, params.N_points)
    logging.info('Load HIT data')
    HIT_data = load_HIT_data()
    logging.info('Filter HIT data')
    LES_data = filter3d(data=HIT_data, scale_k=params.LES_scale, dx=dx, N_points=params.N_points)
    TEST_data = None
    if params.TEST_scale:
        TEST_data = filter3d(data=HIT_data, scale_k=params.TEST_scale, dx=dx, N_points=params.N_points)

    LES_delta = 1 / params.LES_scale
    LES = data.Data(LES_data, LES_delta, params.HOMOGENEOUS, dx)
    TEST = None
    if params.TEST_scale:
        TEST_delta = 1 / params.TEST_scale
        TEST = data.Data(TEST_data, TEST_delta, params.HOMOGENEOUS, dx)

    if spectra:
        logging.info('Calculate LES spectra')
        spectral_density([LES_data['u'], LES_data['v'], LES_data['w']], dx, params.N_points, folder + 'LES')
        if params.TEST_scale:
            logging.info('Calculate TEST spectra')
            spectral_density([TEST_data['u'], TEST_data['v'], TEST_data['w']], dx, params.N_points, folder + 'test')

    logging.info('Plot initial data info')
    if spectra:
        plotting.spectra()
    map_bounds = np.linspace(np.min(LES_data['v'][:, :, 127]), np.max(LES_data['v'][:, :, 127]), 9)
    plotting.compare_filter_fields(HIT_data, LES_data, TEST_data, map_bounds, folder=folder)
    plotting.vel_fields(HIT_data, scale='DNS', map_bounds=map_bounds, folder=folder)
    plotting.vel_fields(LES_data, scale='LES', map_bounds=map_bounds, folder=folder)
    if params.TEST_scale:
        plotting.vel_fields(TEST_data, scale='TEST', map_bounds=map_bounds, folder=folder)
        plotting.sigma_field(TEST.tau_true, scale='TEST', folder=folder)
    plotting.sigma_field(LES.tau_true, scale='LES', folder=folder)
    plotting.sigma_pdf(LES, TEST, bins=params.bins, domain=params.domain, folder=folder)


if __name__ == '__main__':
    main()

