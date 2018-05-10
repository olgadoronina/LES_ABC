import logging
import global_var as g
import numpy as np
import scipy as sp
import scipy.stats
from numpy.fft import fftfreq, fftn, ifftn
from time import time

import abc_class
from sobol_seq import i4_sobol_generate
import itertools


def timer(start, end, label):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    logging.info("{:0>1}:{:0>2}:{:05.2f} \t {}".format(int(hours), int(minutes), seconds, label))



def get_prior(x):
    d = (g.C_limits[:, 1]-g.C_limits[:, 0]) / g.N.each
    ind = np.floor_divide((x - g.C_limits[:, 0]), d)   # find nearest point
    ind = tuple(ind.astype(np.int8, copy=False))
    y = g.prior[ind]
    return y


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
    pdf, _ = np.histogram(array, bins=bins, range=range, normed=1)
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
def sampling_initial_for_MCMC():
    """ Find starting points for MCMC. (Sample randomly and save if distance < eps)
    :return: list of lists of parameters
    """
    C_array = []

    while len(C_array) <= g.N.proc:
        c = np.random.uniform(g.C_limits[:, 0], g.C_limits[:, 1])
        c_start = abc_class.work_function_single_value(list(c))
        if c_start:
            C_array.append(c_start[:-1])
            logging.info('C_start = {}'.format(c_start[:-1]))

    return C_array


def sampling_sobol():
    """ Generate Sobol' sequense of parameters. (low-discrepency quasi-random sampling)
    :return: list of lists of sampled parameters
    """
    C_array = i4_sobol_generate(g.N.params, g.N.calibration)
    for i in range(g.N.params):
        C_array[:, i] = C_array[:, i] * (g.C_limits[i, 1] - g.C_limits[i, 0]) + g.C_limits[i, 0]
    C_array = C_array.tolist()
    return C_array


def sampling_random():
    """ Generate Sobol' sequense of parameters. (low-discrepency quasi-random sampling)
    :return: list of lists of sampled parameters
    """
    C_array = np.random.random(size=(g.N.calibration, g.N.params))
    for i in range(g.N.params):
        C_array[:, i] = C_array[:, i] * (g.C_limits[i, 1] - g.C_limits[i, 0]) + g.C_limits[i, 0]
    C_array = C_array.tolist()
    return C_array


def sampling_uniform_grid():
    """ Create list of lists of N parameters manually (make grid) uniformly distributed on given interval
    :return: list of lists of sampled parameters
    """
    if g.N.params == 1:
        C1 = uniform_grid(g.C_limits[0], g.N.each)
        C_array = []
        for i in C1:
            C_array.append([i])
    else:
        C = np.ndarray((g.N.params - g.N.params_in_task, g.N.each))
        for i in range(g.N.params - g.N.params_in_task):
            C[i, :] = uniform_grid(g.C_limits[i], g.N.each)
        permutation = itertools.product(*C)
        C_array = list(map(list, permutation))
        print(np.array(C_array).shape)
    logging.debug('Form C_array manually: {} samples\n'.format(len(C_array)))
    return C_array


########################################################################################################################
## Distance functions
########################################################################################################################
def distance_between_pdf_KL(pdf_modeled, key, axis=1):
    """Calculate statistical distance between two pdf as
    the Kullback-Leibler (KL) divergence (no symmetry).
    Function for N_params_in_task > 0
    :param pdf_modeled: array of modeled pdf
    :param key: tensor component(key of dict)
    :return: 1D array of calculated distance
    """

    log_modeled = utils.take_safe_log(pdf_modeled)
    dist = np.sum(np.multiply(g.TEST_sp.tau_pdf_true[key], (g.TEST_sp.log_tau_pdf_true[key] - log_modeled)), axis=axis)

    return dist


def distance_between_pdf_L1log(pdf_modeled, key, axis=1):
    """Calculate statistical distance between two pdf as
    :param pdf_modeled: array of modeled pdf
    :return:            scalar of calculated distance
    """

    log_modeled = utils.take_safe_log(pdf_modeled)
    dist = 0.5 * np.sum(np.abs(log_modeled - g.TEST_sp.log_tau_pdf_true[key]), axis=axis)
    return dist


def distance_between_pdf_LSE(pdf_modeled, key, axis=1):
    """ Calculate statistical distance between two pdf as mean((P1-P2)^2).
    :param pdf_modeled: array of modeled pdf
    :param key: tensor component(key of dict)
    :param axis: equal 1 when pdf_modeled is 2D array
    :return: scalar or 1D array of calculated distance
    """
    dist = np.mean((pdf_modeled - g.TEST_sp.tau_pdf_true[key]) ** 2, axis=axis)
    return dist


def distance_between_pdf_L2(pdf_modeled, key, axis=1):
    """ Calculate statistical distance between two pdf as sqrt(sum((P1-P2)^2)).
    :param pdf_modeled: array of modeled pdf
    :param key: tensor component(key of dict)
    :param axis: equal 1 when pdf_modeled is 2D array
    :return: scalar or 1D array of calculated distance
    """
    dist = np.sqrt(np.sum((pdf_modeled - g.TEST_sp.tau_pdf_true[key]) ** 2, axis=axis))
    return dist


def distance_between_pdf_LSElog(pdf_modeled, key, axis=1):
    """ Calculate statistical distance between two pdf as mean((ln(P1)-ln(P2))^2).
    :param pdf_modeled: array of modeled pdf
    :param key: tensor component(key of dict)
    :return: 1D array of calculated distance
    """
    log_modeled = utils.take_safe_log(pdf_modeled)
    dist = np.mean((log_modeled - g.TEST_sp.log_tau_pdf_true[key]) ** 2, axis=axis)
    return dist


def distance_between_pdf_L2log(pdf_modeled, key, axis=1):
    """ Calculate statistical distance between two pdf.
    :param pdf_modeled:
    :param key:
    :param axis:
    :return:
    """
    log_modeled = utils.take_safe_log(pdf_modeled)
    dist = np.sum((log_modeled - g.TEST_sp.log_tau_pdf_true[key]) ** 2, axis=axis)
    return dist

