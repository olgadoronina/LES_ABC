from math import *
import numpy as np
import matplotlib.pyplot as plt
from time import time, sleep
import gc
import utils
import matplotlib as mpl
import multiprocessing as mp
import random as rand
import logging


plt.style.use(['seaborn-paper', 'mystyle'])
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

# Parameters of post-ploting
PLOT_ALL_DIST = 0   # Show all distances with unaccepted ones

# Path to dns data
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
C_limits[0] = [0, 0.3]
C_limits[1] = [-0.15, 0.15]
C_limits[2] = [-0.35, 0.2]
C_limits[3] = [-0.2, 0.2]

bins = 100  # for joint pdf
#################################################
# abs algorithm
eps = 500        # acceptance tolerance
N = int(1.5e5)  # number of samples
M = 16          # number of training points
ORDER = 2       # order of eddy-viscosity model
#################################################
# Parallel regime parameters
PARALLEL = 1        # 0 - Not parallel; 1 - parallel
PROGRESSBAR = 0     # 0 - pool.map(no bar); 1 - pool.imap_unordered(progressbar); 2 - pool.map_async(text progress)
N_proc = 12          # Number of processes
