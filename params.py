from math import *
import numpy as np
import matplotlib.pyplot as plt

plt.style.use(['seaborn-paper', 'mystyle'])

datafile_u = './data/HIT_u.bin'
datafile_v = './data/HIT_v.bin'
datafile_w = './data/HIT_w.bin'

# Case parameters
N_points = [256, 256, 256] # number of points
N_dx = [255, 255, 255]  # number of intervals
lx = [2 * pi, 2 * pi, 2 * pi]  # domain size
dx = np.divide(lx, N_points)  # grid cell size (resolution)
# N = Nx[0] * Nx[1] * Nx[2]


LES_scale = 10
TEST_scale = 5

LES_delta = 1/LES_scale
TEST_delta = 1/TEST_scale

