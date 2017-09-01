from ABC.params import *


def timer(start, end, label):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:05.2f}".format(int(minutes), seconds), '\t', label)


def read_data():
    data = dict()
    data['u'] = np.reshape(np.fromfile(datafile_u, dtype=type_of_bin_data), tuple(N_points))
    data['v'] = np.reshape(np.fromfile(datafile_v, dtype=type_of_bin_data), tuple(N_points))
    data['w'] = np.reshape(np.fromfile(datafile_w, dtype=type_of_bin_data), tuple(N_points))
    # to put x index in first place
    for key, value in data.items():
        data[key] = np.swapaxes(value, 0, 2)
    return data


def pdf_from_array_with_x(array, bins, range):
    pdf, edges = np.histogram(array, bins=bins, range=range, normed=1)
    x = (edges[1:] + edges[:-1]) / 2
    return x, pdf


def pdf_from_array(array, bins, range):
    pdf, edges = np.histogram(array, bins=bins, range=range, normed=1)
    return pdf


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
