import numpy.fft as fft
from utils import *
from time import time

def tophat_kernel(k, limit):
    """Create 3D array of Tophat filter3d.
        k - array of wave numbers;
        limit - size of the filter3d."""
    a = np.zeros((len(k[0]), len(k[1]), len(k[2])), dtype=np.float32)
    for indx, kx in enumerate(k[0]):
        for indy, ky in enumerate(k[1]):
            for indz, kz in enumerate(k[2]):
                a[indx, indy, indz] = sqrt(kx ** 2 + ky ** 2 + kz ** 2)

    kernel = np.piecewise(a, [a <= limit, a > limit], [1, 0])
    return kernel
    # Gaussian kernel
    # xaxis = np.linspace(0, lx[0], Nx[0])
    # yaxis = np.linspace(0, lx[1], Nx[1])
    # zaxis = np.linspace(0, lx[2], Nx[2])
    # x = xaxis[:, None, None]
    # y = yaxis[None, :, None]
    # z = zaxis[None, None, :]
    # return np.exp(-(x**2 + y**2 + z**2)/(2.0*sigma_square))/(np.sqrt(2.0*np.pi*sigma_square))


def filter3d(data, scale_k, filename=None):
    """ Tophat filter in Fourier space for dictionary of 3D arrays.
        data - dictionary of numpy arrays;
        scale_k - wave number, which define size of filter."""
    # FFT
    start = time()
    FFT = dict()
    for key, value in data.items():
        FFT[key] = fft.fftn(value)
    k = [fft.fftfreq(N_points[0], dx[0]), fft.fftfreq(N_points[1], dx[1]), fft.fftfreq(N_points[2], dx[2])]
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
        result[key] = fft.ifftn(value).real
    end = time()
    timer(start, end, 'Time for iFFT')

    fft_filtered.clear()

    if filename:
        print('\nWrite file in ./data/' + filename + '.npz')
        file = './data/' + filename + '.npz'
        np.savez(file, **result)

    return result

def filter3d_array(array, scale_k):

    fft_array = fft.fftn(array)
    k = [fft.fftfreq(N_points[0], dx[0]), fft.fftfreq(N_points[1], dx[1]), fft.fftfreq(N_points[2], dx[2])]
    kernel = tophat_kernel(k, scale_k)
    fft_filtered = np.multiply(fft_array, kernel)
    result = fft.ifftn(fft_filtered).real

    return result
