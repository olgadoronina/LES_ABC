from math import *
import matplotlib as mpl
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
PLOT_INIT_INFO = 0

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
ORDER = 2           # order of eddy-viscosity model
N_params_force = 3
########################################################################################################################
# Sampling
sampling = 'uniform'    # 'uniform', 'random', 'sobol' , 'MCMC'
N_each = 10
N_params_in_task = 0  # only 0, 1 or 2  #only 0 and 2 for calibration
########################################################################################################################
# abs algorithm
bins = 100  # for pdf comparison
# domain = [-0.45, 0.45]  # for pdf comparison
domain = [-0.7, 0.7]  # for pdf comparison
num_bin_joint = 10
eps = 5000   # acceptance tolerance
########################################################################################################################
MCMC = 2    # 1 = MCMC; 2 = IMCMC
N_total = 10**5
N_calibration = 10**3  # recommended 10^p, where p is number of params
PMC = 0
#######################################################################################################################
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
# C_limits[0] = [0.0, 0.30]
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
var = np.empty(6)
var[0] = (-2*C_limits[0][0]**2 - (-2*C_limits[0][1]**2)) / 20
var[1] = (C_limits[1][1] - C_limits[1][0]) / 20
var[2] = (C_limits[2][1] - C_limits[2][0]) / 20
var[3] = (-2*C_limits[0][0]**2 - (-2*C_limits[0][1]**2)) / 20
var[4] = (C_limits[1][1] - C_limits[1][0]) / 20
var[5] = (C_limits[2][1] - C_limits[2][0]) / 20

########################################################################################################################
# Parallel regime parameters
PROGRESSBAR = 1    # 0 - pool.map(no bar); 1 - pool.imap_unordered(progressbar); 2 - pool.map_async(text progress)
N_proc = 6          # Number of processes
########################################################################################################################


mpl.rcParams['font.size'] = 10
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rc('text', usetex=True)
mpl.rcParams['axes.labelsize'] = mpl.rcParams['font.size']
mpl.rcParams['axes.titlesize'] = 1.5 * mpl.rcParams['font.size']
mpl.rcParams['legend.fontsize'] = mpl.rcParams['font.size']
mpl.rcParams['xtick.labelsize'] = mpl.rcParams['font.size']
mpl.rcParams['ytick.labelsize'] = mpl.rcParams['font.size']
# plt.rcParams['savefig.dpi'] = 2 * plt.rcParams['savefig.dpi']
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.width'] = 1
# mpl.rcParams['legend.frameon'] = False
# plt.rcParams['legend.loc'] = 'center left'
mpl.rcParams['axes.linewidth'] = 1