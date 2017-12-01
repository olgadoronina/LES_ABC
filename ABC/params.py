from math import *

import numpy as np

# import cProfile

# prof = cProfile.Profile()
########################################################################################################################

########################################################################################################################
# Path to data
LOAD = 1          # Load filtered data or filter from DNS
DATA = 'JHU_data'
data_folder = './data_input/JohnHopkins/'
# DATA = 'CU_data'
# data_folder = './data_input/HIT_DNS_N256/'
########################################################################################################################
# Initial case parameters
HOMOGENEOUS = 1    # Use symmetry of tau tensor
N_point = 256
N_points = [N_point, N_point, N_point]      # number of points
lx = [2 * pi, 2 * pi, 2 * pi]   # domain size
# Filter scales
# LES_scale = 10
# TEST_scale = 5
LES_scale = 30/2/pi
TEST_scale = 15/2/pi
########################################################################################################################
# abs algorithm
bins = 100  # for pdf
domain = [-1.1, 1.1]  # for pdf comparison

num_bin_joint = 20

N_each = 1000
N_params_in_task = 0  # only 0, 1 or 2

M = 64          # number of training points
ORDER = 1      # order of eddy-viscosity model
N_params_force = 0
eps = 170     # acceptance tolerance
########################################################################################################################

# Sample limits
C_limits = np.zeros((10, 2))
# # for 4 params:
# C_limits[0] = [0.0, 0.25]
# C_limits[1] = [-0.2, 0.2]
# C_limits[2] = [-0.2, 0.2]
# C_limits[3] = [-0.2, 0.2]
#
# C_limits[4] = [-0.3, 0.3]
# C_limits[5] = [-0.3, 0.3]

# best
C_limits[0] = [0.15, 0.30]
# C_limits[0] = [0.0, 0.25]
C_limits[1] = [-0.15, 0.15]
C_limits[2] = [-0.15, 0.15]
C_limits[3] = [-0.15, 0.15]

C_limits[4] = [-0.3, 0.3]
C_limits[5] = [-0.15, 0.15]
params_names = [r'$C_s$', r'$C_2$', r'$C_3$', r'$C_4$', r'$C_5$', r'$C_6$', r'$C_7$', r'$C_8$', r'$C_9$']
########################################################################################################################
# Parallel regime parameters
PROGRESSBAR = 1     # 0 - pool.map(no bar); 1 - pool.imap_unordered(progressbar); 2 - pool.map_async(text progress)
N_proc = 4          # Number of processes
########################################################################################################################
