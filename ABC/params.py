from math import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from time import time, sleep
import gc
import random as rand
import logging

########################################################################################################################
# mpl.style.use(['dark_background','mystyle'])
# mpl.style.use(['mystyle'])
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

########################################################################################################################
# Parameters of post-plotting
PLOT_ALL_DIST = 0   # Show all distances with unaccepted ones

########################################################################################################################
# Path to data
LOAD = 0            # Load filtered data or filter from DNS
loadfile_LES = '../data_input/HIT_DNS_N256/LES.npz'
loadfile_TEST = '../data_input/HIT_DNS_N256/TEST.npz'
datafile_u = '../data_input/HIT_DNS_N256/Velocity1_003.rst'
datafile_v = '../data_input/HIT_DNS_N256/Velocity2_003.rst'
datafile_w = '../data_input/HIT_DNS_N256/Velocity3_003.rst'
type_of_bin_data = np.float64

########################################################################################################################
# Initial case parameters
HOMOGENEOUS = 1    # Use symmetry of tau tensor
N_point = 256
N_points = [N_point, N_point, N_point]      # number of points
lx = [2 * pi, 2 * pi, 2 * pi]   # domain size
dx = np.divide(lx, N_points)
# Filter scales
LES_scale = 10
TEST_scale = 5
# Characteristic length \Delta
LES_delta = 1/LES_scale
TEST_delta = 1/TEST_scale

########################################################################################################################
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

########################################################################################################################
# abs algorithm
eps = 50        # acceptance tolerance
N = int(1e6)    # number of samples
M = 64          # number of training points
ORDER = 1       # order of eddy-viscosity model
USE_C4 = 0

########################################################################################################################
# Parallel regime parameters
PARALLEL = 1        # 0 - Not parallel; 1 - parallel
PROGRESSBAR = 0    # 0 - pool.map(no bar); 1 - pool.imap_unordered(progressbar); 2 - pool.map_async(text progress)
N_proc = 4          # Number of processes

########################################################################################################################
