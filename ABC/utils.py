import logging
import global_var as g
import numpy as np
import scipy as sp
import scipy.stats
from numpy.fft import fftfreq, fftn


def timer(start, end, label):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    logging.info("{:0>2}:{:05.2f} \t {}".format(int(minutes), seconds, label))



def pdf_from_array_with_x(array, bins, range):
    pdf, edges = np.histogram(array, bins=bins, range=range, normed=1)
    x = (edges[1:] + edges[:-1]) / 2
    return x, pdf


def pdf_from_array_improved(array, bins, domain, N_each):
    pdf = np.empty((N_each, bins))
    for i in range(N_each):
        pdf[i, :] = np.histogram(array[i, :], bins=bins, range=domain, normed=1)[0]
    return pdf


def pdf_from_array(array, bins, range):
    pdf, edges = np.histogram(array, bins=bins, range=range, normed=1)
    return pdf


def baseconvert(x, newbase, number_digits):
    """Converts given number x, from base 10 to base 'newbase'
    x -- the number in base 10
    newbase -- base to convert
    number_digits -- number of digits in new base (add zero in the beginning)
    """
    assert(x >= 0)
    r = []
    while x > 0:
        r = [x % newbase] + r
        x //= newbase
    for i in range(number_digits-len(r)):
        r = [0] + r
    return r


def shell_average(spect3D, N_point, k_3d):
    """ Compute the 1D, shell-averaged, spectrum of the 3D Fourier-space
    variable E3.
    :param E3: 3-dimensional complex or real Fourier-space scalar
    :param km:  wavemode of each n-D wavevector
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


def uniform_grid(C_limits, N_each):
    C_tmp = np.linspace(C_limits[0], C_limits[1], N_each + 1)
    C_tmp = C_tmp[:-1] + (C_tmp[1] - C_tmp[0]) / 2
    return C_tmp


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, s = np.mean(a), np.std(a)
    h = s / np.sqrt(n) * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, h

def normalize_params(C):
    """
    Normalize given parameters to -1<C<1 hypercube.
    :param C: given parameters
    :return: normalized parameters
    """
    return 2*(C - g.C_limits[:, 0])/(g.C_limits[:, 1] - g.C_limits[:, 0]) - 1

def unnormalize_params(C):
    return 0.5*(C+1)*(g.C_limits[:, 1] - g.C_limits[:, 0]) + g.C_limits[:, 0]


# import scipy.ndimage as ndimage
#
# # Other filters
# for example
#     for key, value in HIT.items():
#         LES[key] = ndimage.gaussian_filter(value, sigma=3)
#         TEST[key] = ndimage.gaussian_filter(value, sigma=5)
# map_bounds = np.linspace(np.min(HIT['u'][:, :, 127]), np.max(HIT['u'][:, :, 127]), 20)
# imagesc([HIT['u'][:, :, 127], LES['u'][:, :, 127], TEST['u'][:, :, 127]], map_bounds, 'gaussian_filter')
#
# for key, value in HIT.items():
#     LES[key] = ndimage.uniform_filter(value, size=5, mode='constant')
#     TEST[key] = ndimage.uniform_filter(value, size=10, mode='constant')
# map_bounds = np.linspace(np.min(HIT['u'][:, :, 127]), np.max(HIT['u'][:, :, 127]), 20)
# imagesc([HIT['u'][:, :, 127], LES['u'][:, :, 127], TEST['u'][:, :, 127]], map_bounds, 'Physical_sharp')
#
# for key, value in HIT.items():
#     input = np.fft.fftn(value)
#     LES[key] = np.fft.ifftn(ndimage.fourier_uniform(input, size=5))
#     TEST[key] = np.fft.ifftn(ndimage.fourier_uniform(input, size=10))
# map_bounds = np.linspace(np.min(HIT['u'][:, :, 127]), np.max(HIT['u'][:, :, 127]), 20)
# imagesc([HIT['u'][:, :, 127], LES['u'].real[:, :, 127], TEST['u'].real[:, :, 127]], map_bounds, 'Fourier_sharp')


