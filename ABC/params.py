from math import *
import numpy as np

# import cProfile
# prof = cProfile.Profile()
########################################################################################################################
# Path to data
LOAD = 1          # Load filtered data or filter from DNS
DATA = 'JHU_data'
data_folder = './data_input/JohnHopkins/'
# DATA = 'CU_data'
# data_folder = './data_input/HIT_DNS_N256/'
# Plotting
plot_folder = './plots/'
########################################################################################################################
# Initial case parameters
N_point = 256
N_points = [N_point, N_point, N_point]      # number of points
lx = [2 * pi, 2 * pi, 2 * pi]               # domain size
# Filter scales
# LES_scale = 10
# TEST_scale = 5
LES_scale = 30/2/np.pi
TEST_scale = None
M = 64                                     # number of training points (sparse data)
########################################################################################################################
# Model parameters
HOMOGENEOUS = 1     # Use symmetry of tau tensor
ORDER = 3           # order of eddy-viscosity model
N_params_force = 6
########################################################################################################################
# Sampling
sampling = 'uniform'    # 'uniform', 'random', 'sobol' , 'MCMC'
N_each = 100
N_params_in_task = 2  # only 0, 1 or 2  #only 0 and 2 for calibration
########################################################################################################################
# abs algorithm
bins = 100  # for pdf comparison
domain = [-0.45, 0.45]  # for pdf comparison
# domain = [-0.7, 0.7]  # for pdf comparison
num_bin_joint = 20
eps = 2000   # acceptance tolerance
########################################################################################################################
MCMC = 2    # 1 = MCMC; 2 = IMCMC
N_total = 10**7
########################################################################################################################
N_calibration = 0  # recommended 10^p, where p is number of params
x = 0.01   # percent of accepted for calibration step
phi = 1
########################################################################################################################
# Gaussian mixture
N_gaussians = 30  # number of gaussians
########################################################################################################################
PMC = 0
#######################################################################################################################
# Sample limits
C_limits = np.zeros((10, 2))
C_limits[0] = [-0.3, 0.3]
C_limits[1] = [-0.5, 0.5]
C_limits[2] = [-0.2, 0.2]
C_limits[3] = [-0.2, 0.2]
C_limits[4] = [-1, 1]
C_limits[5] = [-0.5, 0.5]
C_limits[6] = [-1, 1]
C_limits[7] = [-1, 1]
C_limits[8] = [-1, 1]
C_limits[9] = [-1, 1]


################################
var = np.empty(10)
var = (C_limits[:, 1] - C_limits[:, 0])/2
########################################################################################################################
# Parallel regime parameters
PROGRESSBAR = 1    # 0 - pool.map(no bar); 1 - pool.imap_unordered(progressbar); 2 - pool.map_async(text progress)
N_proc = 6          # Number of processes
########################################################################################################################
