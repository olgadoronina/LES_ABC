import scipy.ndimage as ndimage

# Other filters
for example
    for key, value in HIT.items():
        LES[key] = ndimage.gaussian_filter(value, sigma=3)
        TEST[key] = ndimage.gaussian_filter(value, sigma=5)
map_bounds = np.linspace(np.min(HIT['u'][:, :, 127]), np.max(HIT['u'][:, :, 127]), 20)
imagesc([HIT['u'][:, :, 127], LES['u'][:, :, 127], TEST['u'][:, :, 127]], map_bounds, 'gaussian_filter')

for key, value in HIT.items():
    LES[key] = ndimage.uniform_filter(value, size=5, mode='constant')
    TEST[key] = ndimage.uniform_filter(value, size=10, mode='constant')
map_bounds = np.linspace(np.min(HIT['u'][:, :, 127]), np.max(HIT['u'][:, :, 127]), 20)
imagesc([HIT['u'][:, :, 127], LES['u'][:, :, 127], TEST['u'][:, :, 127]], map_bounds, 'Physical_sharp')

for key, value in HIT.items():
    input = np.fft.fftn(value)
    LES[key] = np.fft.ifftn(ndimage.fourier_uniform(input, size=5))
    TEST[key] = np.fft.ifftn(ndimage.fourier_uniform(input, size=10))
map_bounds = np.linspace(np.min(HIT['u'][:, :, 127]), np.max(HIT['u'][:, :, 127]), 20)
imagesc([HIT['u'][:, :, 127], LES['u'].real[:, :, 127], TEST['u'].real[:, :, 127]], map_bounds, 'Fourier_sharp')
