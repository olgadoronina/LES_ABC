from math import *

import numpy as np

# Case parameters
Nx = [256, 256, 256]  # grid dimensions
lx = [2 * pi, 2 * pi, 2 * pi]  # domain size
dx = np.divide(lx, Nx)  # grid cell size (resolution)
N = Nx[0] * Nx[1] * Nx[2]
