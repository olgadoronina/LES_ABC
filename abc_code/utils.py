import logging
import abc_code.global_var as g
import numpy as np
import scipy as sp
import scipy.stats
from numpy.fft import fftfreq, fftn, ifftn
from time import time

from abc_code.sobol_seq import i4_sobol_generate
from fast_histogram import histogram1d
import abc_code.distance as dist
import itertools


def timer(start, end, label):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    logging.info("{:0>1}:{:0>2}:{:05.2f} \t {}".format(int(hours), int(minutes), seconds, label))


def pdf_from_array_with_x(array, bins, range):
    pdf, edges = np.histogram(array, bins=bins, range=range, normed=1)
    x = (edges[1:] + edges[:-1]) / 2
    return x, pdf


def pdf_from_array_improved(array, bins, domain, N_each):
    pdf = np.empty((N_each, bins))
    for i in range(N_each):
        pdf[i, :] = np.histogram(array[i, :], bins=bins, range=domain, normed=1)[0]
    return pdf


def pdf_from_array_np(array, bins, range):
    pdf, _ = np.histogram(array, bins=bins, range=range, normed=1)
    return pdf


def pdf_from_array(array, bins, range):
    pdf = histogram1d(array.flatten(), bins=bins, range=range)
    norm = np.sum(pdf)/bins
    return pdf/norm


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


def take_safe_log(x):
    """Takes natural logarithm and put g.TINY number where x = 0"""
    log_fill = np.empty_like(x)
    log_fill.fill(g.TINY_log)
    log = np.log(x, out=log_fill, where=x > g.TINY)
    return log

def covariance_recursive(x, t, cov_prev, mean_prev, s_d):
    mean_new = t / (t + 1) * mean_prev + 1 / (t + 1) * x
    cov = (t - 1) / t * cov_prev + \
          s_d / t * (t * np.outer(mean_prev, mean_prev) - (t + 1) * np.outer(mean_new, mean_new) + np.outer(x, x))
    return cov, mean_new


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


def filter3d(data, scale_k, dx, N_points, filename=None):
    """ Tophat filter in Fourier space for dictionary of 3D arrays.
        data - dictionary of numpy arrays;
        scale_k - wave number, which define size of filter."""
    # FFT
    start = time()
    FFT = dict()
    for key, value in data.items():
        FFT[key] = fftn(value)
    k = [fftfreq(N_points[0], dx[0]), fftfreq(N_points[1], dx[1]), fftfreq(N_points[2], dx[2])]
    end = time()
    timer(start, end, 'Time for FFT')

    # Filtering
    start = time()
    kernel = tophat_kernel(k, scale_k)
    end = time()
    timer(start, end, 'Time for creating filter kernel')

    start = time()
    result = dict()
    fft_filtered = dict()
    for key, value in FFT.items():
        fft_filtered[key] = np.multiply(value, kernel)
    end = time()
    timer(start, end, 'Time for filtering')

    FFT.clear()

    start = time()
    for key, value in fft_filtered.items():
        result[key] = ifftn(value).real
    end = time()
    timer(start, end, 'Time for iFFT')

    fft_filtered.clear()

    if filename:
        logging.info('\nWrite file in ./data/' + filename + '.npz')
        file = './data/' + filename + '.npz'
        np.savez(file, **result)

    return result

# def filter3d_array(array, scale_k):
#
#     fft_array = fftn(array)
#     k = [fftfreq(N_points[0], dx[0]), fftfreq(N_points[1], dx[1]), fftfreq(N_points[2], dx[2])]
#     kernel = tophat_kernel(k, scale_k)
#     fft_filtered = np.multiply(fft_array, kernel)
#     result = ifftn(fft_filtered).real
#
#     return result
#
# def filter3d_array_inFspace(array, scale_k):
#     logging.info(array.shape)
#     k = [fftfreq(N_points[0], dx[0]), fftfreq(N_points[1], dx[1]), fftfreq(N_points[2], dx[2])]
#     kernel = tophat_kernel(k, scale_k)
#     fft_filtered = np.multiply(array, kernel)
#
#     return fft_filtered


########################################################################################################################
## Sampling functions
########################################################################################################################
def sampling_initial_for_MCMC(N_proc, C_limits, eps):
    """ Find starting points for MCMC. (Sample randomly and save if distance < eps)
    :return: list of lists of parameters
    """
    C_array = []
    while len(C_array) <= N_proc:
        c = np.random.uniform(C_limits[:, 0], C_limits[:, 1])
        d = dist.calc_dist(c)
        if d <= eps:
            C_array.append(c)
            logging.info('C_start = {}'.format(c))

    return C_array


def sampling_initial_for_gaussian_mixture(N_proc, N_gaussians, C_limits, eps):
    """ Find starting points for Gaussian Mixture. (Sample randomly and save if distance < eps)
    :return: list of lists of parameters
    """
    C_array = []
    start = time()
    from tqdm import tqdm
    with tqdm(total=N_proc*N_gaussians) as pbar:
        for i in range(N_proc):
            c_array = []
            while len(c_array) < N_gaussians:
                c = np.random.uniform(C_limits[:, 0], C_limits[:, 1])
                dist = calc_dist(c)
                if dist <= eps:
                    c_array.append(c)
                    pbar.update()
            C_array.append(np.array(c_array))
        pbar.close()
    end = time()
    timer(start, end, 'Time for sampling')
    return C_array


def sampling_sobol(N_total, C_limits):
    """ Generate Sobol' sequense of parameters. (low-discrepency quasi-random sampling)
    :return: list of lists of sampled parameters
    """
    N_params = len(C_limits)
    C_array = i4_sobol_generate(N_params, N_total)
    for i in range(N_params):
        C_array[:, i] = C_array[:, i] * (C_limits[i, 1] - C_limits[i, 0]) + C_limits[i, 0]
    C_array = C_array.tolist()
    return C_array


def sampling_random(N_total, C_limits):
    """ Generate Sobol' sequense of parameters. (low-discrepency quasi-random sampling)
    :return: list of lists of sampled parameters
    """
    N_params = len(C_limits)
    C_array = np.random.random(size=(N_total, N_params))
    for i in range(g.N.params):
        C_array[:, i] = C_array[:, i] * (C_limits[i, 1] - C_limits[i, 0]) + C_limits[i, 0]
    C_array = C_array.tolist()
    return C_array


def sampling_uniform_grid(N_each, N_params_in_task, C_limits):
    """ Create list of lists of N parameters manually (make grid) uniformly distributed on given interval
    :return: list of lists of sampled parameters
    """
    N_params = len(C_limits)
    if N_params == 1:
        # C1 = np.linspace(C_limits[0, 0], C_limits[0, 1], N_each)
        C1 = uniform_grid(C_limits[0], N_each)
        C_array = []
        for i in C1:
            C_array.append([i])
    else:
        C = np.empty((N_params - N_params_in_task, N_each))
        for i in range(N_params - N_params_in_task):
            # C[i, :] = np.linspace(C_limits[i, 0], C_limits[i, 1], N_each)
            C[i, :] = uniform_grid(C_limits[i], N_each)
        permutation = itertools.product(*C)
        C_array = list(map(list, permutation))
    logging.debug('Form C_array as uniform grid: {} samples\n'.format(len(C_array)))
    return C_array




