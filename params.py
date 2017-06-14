from math import *
import numpy as np
import matplotlib.pyplot as plt
from time import time
import gc
import utils
import matplotlib as mpl
import multiprocessing as mp
import random as rand

# plt.style.use(['seaborn-paper', 'mystyle'])

PARALLEL = 0

# path to dns data
datafile_u = './data/HIT_u.bin'
datafile_v = './data/HIT_v.bin'
datafile_w = './data/HIT_w.bin'

# Case parameters
N_points = [256, 256, 256]  # number of points
N_dx = [255, 255, 255]  # number of intervals
lx = [2 * pi, 2 * pi, 2 * pi]  # domain size
dx = np.divide(lx, N_points)  # grid cell size (resolution)
# N = Nx[0] * Nx[1] * Nx[2]

# Scales
LES_scale = 10
TEST_scale = 5
LES_delta = 1/LES_scale
TEST_delta = 1/TEST_scale


# params for abc algorithm
domain = [-1.1, 1.1]
Cs_limits = [0.19, 0.23]
bins = 100
eps = 50
N = 24
