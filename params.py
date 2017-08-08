from math import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from time import time, sleep
import gc
import utils
import random as rand
import logging

mpl.style.use(['dark_background','mystyle'])
mpl.style.use(['mystyle'])
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

# Parameters of post-plotting
PLOT_ALL_DIST = 0   # Show all distances with unaccepted ones

# Path to data
LOAD = 1
loadfile_LES = './data/LES.npz'
loadfile_TEST = './data/TEST.npz'
datafile_u = './data/HIT_u.bin'
datafile_v = './data/HIT_v.bin'
datafile_w = './data/HIT_w.bin'

# Initial case parameters
HOMOGENEOUS = 1
N_points = [256, 256, 256]      # number of points
lx = [2 * pi, 2 * pi, 2 * pi]   # domain size
dx = np.divide(lx, N_points)
# Filter scales
LES_scale = 10
TEST_scale = 5
# Characteristic length \Delta
LES_delta = 1/LES_scale
TEST_delta = 1/TEST_scale

# Params for abc algorithm
domain = [-1.1, 1.1]  # for pdf comparison
# Sample limits
C_limits = np.zeros((4, 2))
C_limits[0] = [0.1, 0.3]
C_limits[1] = [-0.2, 0.2]
C_limits[2] = [-0.15, 0.05]
C_limits[3] = [-0.15, 0.2]
bins = 100  # for pdf
num_bin_joint = 20
params_names = [r'$C_s$', r'$C_2$', r'$C_3$', r'$C_4$']
#################################################
# abs algorithm
eps = 50        # acceptance tolerance
N = int(1e6)    # number of samples
M = 64          # number of training points
ORDER = 2       # order of eddy-viscosity model
USE_C4 = 0
#################################################
# Parallel regime parameters
PARALLEL = 1        # 0 - Not parallel; 1 - parallel
PROGRESSBAR = 1    # 0 - pool.map(no bar); 1 - pool.imap_unordered(progressbar); 2 - pool.map_async(text progress)
N_proc = 6          # Number of processes
