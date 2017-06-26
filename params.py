from math import *
import numpy as np
import matplotlib.pyplot as plt
from time import time
import gc
import utils
import matplotlib as mpl
import multiprocessing as mp
import random as rand
import logging

plt.style.use(['seaborn-paper', 'mystyle'])
logging.basicConfig(format='%(levelname)s: %(message)s',level=logging.INFO)

PARALLEL = 1

PLOT_ALL_DIST = 0

# path to dns data
datafile_u = './data/HIT_u.bin'
datafile_v = './data/HIT_v.bin'
datafile_w = './data/HIT_w.bin'



# Case parameters
N_points = [256, 256, 256]  # number of points
M = 64                      # number of training points
lx = [2 * pi, 2 * pi, 2 * pi]  # domain size
# dx = np.divide(lx, N_points)  # grid cell size (resolution)
# N = Nx[0] * Nx[1] * Nx[2]

# Scales
LES_scale = 10
TEST_scale = 5

# HIT_delta = lx[0]/N_points[0]
LES_delta = 1/LES_scale
TEST_delta = 1/TEST_scale

# params for abc algorithm
domain = [-1.1, 1.1]
C_limits = np.zeros((4, 2))
C_limits[0] = [0, 0.3]
C_limits[1] = [-0.15, 0.15]
C_limits[2] = [-0.35, 0.2]
C_limits[3] = [-0.2, 0.2]

bins = 100
eps = 50
N = 10000
M = 64


