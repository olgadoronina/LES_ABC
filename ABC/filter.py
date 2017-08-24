import numpy.fft as fft
from ABC.utils import timer
from ABC.params import *
from numpy.fft import fftfreq, fft, fftn, ifftn

# try: # need to check if there is fftn in pyfftw
#     from pyfftw.interfaces.numpy_fft import fft, ifft, irfft2, rfft2
#     import pyfftw
#     pyfftw.interfaces.cache.enable()
# except ImportError:
#     pass    # Rely on numpy.fft

def tophat_kernel(k, limit):
    """Create 3D array of Tophat filter.
        k - array of wave numbers;
        limit - size of the filter3d."""
    a = np.zeros((len(k[0]), len(k[1]), len(k[2])), dtype=np.float32)
    for indx, kx in enumerate(k[0]):
        for indy, ky in enumerate(k[1]):
            for indz, kz in enumerate(k[2]):
                a[indx, indy, indz] = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)

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
        print('\nWrite file in ./data/' + filename + '.npz')
        file = './data/' + filename + '.npz'
        np.savez(file, **result)

    return result

def filter3d_array(array, scale_k):

    fft_array = fftn(array)
    k = [fftfreq(N_points[0], dx[0]), fftfreq(N_points[1], dx[1]), fftfreq(N_points[2], dx[2])]
    kernel = tophat_kernel(k, scale_k)
    fft_filtered = np.multiply(fft_array, kernel)
    result = ifftn(fft_filtered).real

    return result

def filter3d_array_inFspace(array, scale_k):

    print(array.shape)
    k = [fftfreq(N_points[0], dx[0]), fftfreq(N_points[1], dx[1]), fftfreq(N_points[2], dx[2])]
    kernel = tophat_kernel(k, scale_k)
    fft_filtered = np.multiply(array, kernel)

    return fft_filtered