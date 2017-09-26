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
mpl.style.use(['mystyle'])
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

########################################################################################################################
# Parameters of post-plotting
PLOT_ALL_DIST = 0   # Show all distances with unaccepted ones

########################################################################################################################
# Path to data
LOAD = 1          # Load filtered data or filter from DNS
# ## JHU data
data_folder = './data_input/JohnHopkins/'
datafile_u = data_folder + 'HIT_u.bin'
datafile_v = data_folder + 'HIT_v.bin'
datafile_w = data_folder + 'HIT_w.bin'
type_of_bin_data = np.float32
# # CU data
# data_folder = './data_input/HIT_DNS_N256/'
# datafile_u = data_folder + 'Velocity1_003.rst'
# datafile_v = data_folder + 'Velocity2_003.rst'
# datafile_w = data_folder + 'Velocity3_003.rst'
# type_of_bin_data = np.float64

loadfile_LES = data_folder + 'LES.npz'
loadfile_TEST = data_folder + 'TEST.npz'
########################################################################################################################
# Initial case parameters
HOMOGENEOUS = 1    # Use symmetry of tau tensor
N_point = 256
N_points = [N_point, N_point, N_point]      # number of points
lx = [2 * pi, 2 * pi, 2 * pi]   # domain size
dx = np.divide(lx, N_points)
# Filter scales
LES_scale = 30/2/pi
TEST_scale = 15/2/pi
# Characteristic length \Delta
LES_delta = 1/LES_scale
TEST_delta = 1/TEST_scale

########################################################################################################################
# abs algorithm
bins = 100  # for pdf
num_bin_joint = 20
N_each = 10
N_params = 6
N_params_in_pool = 4


M = 64          # number of training points
ORDER = 3       # order of eddy-viscosity model

USE_C3 = 0
eps = 200      # acceptance tolerance
########################################################################################################################
# Params for abc algorithm
domain = [-2.1, 2.1]  # for pdf comparison
# Sample limits
C_limits = np.zeros((N_params, 2))
C_limits[0] = [0.1, 0.3]
C_limits[1] = [-0.3, 0]
C_limits[2] = [-0.1, 0.2]
C_limits[3] = [-0.15, 0.2]
C_limits[4] = [-0.2, 0.2]
C_limits[5] = [-0.3, 0.3]
params_names = [r'$C_s$', r'$C_2$', r'$C_3$', r'$C_4$', r'$C_5$', r'$C_6$', r'$C_7$', r'$C_8$', r'$C_9$']



########################################################################################################################
# Parallel regime parameters
PARALLEL = 1        # 0 - Not parallel; 1 - parallel
PROGRESSBAR = 1     # 0 - pool.map(no bar); 1 - pool.imap_unordered(progressbar); 2 - pool.map_async(text progress)
N_proc = 3          # Number of processes

########################################################################################################################
