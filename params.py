from math import *
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['seaborn-paper', 'mystyle'])

# Case parameters
Nx = [256, 256, 256]  # grid dimensions
lx = [2 * pi, 2 * pi, 2 * pi]  # domain size
dx = np.divide(lx, Nx)  # grid cell size (resolution)
# N = Nx[0] * Nx[1] * Nx[2]

LES_scale = 10
TEST_scale = 5

LES_delta = 2*pi/LES_scale
TEST_delta = 2*pi/TEST_scale
