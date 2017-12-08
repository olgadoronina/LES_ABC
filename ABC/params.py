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
########################################################################################################################
# Plotting
plot_folder = './plots/'
PLOT_INIT_INFO = 0
########################################################################################################################
# Initial case parameters
HOMOGENEOUS = 0    # Use symmetry of tau tensor
N_point = 256
N_points = [N_point, N_point, N_point]      # number of points
lx = [2 * pi, 2 * pi, 2 * pi]   # domain size
# Filter scales
LES_scale = 10
TEST_scale = 5
# LES_scale = 30/2/pi
# TEST_scale = 15/2/pi
########################################################################################################################
# abs algorithm
bins = 100  # for pdf
domain = [-1.1, 1.1]  # for pdf comparison
num_bin_joint = 20
N_each = 100
N_params_in_task = 0  # only 0, 1 or 2
M = 64          # number of training points
ORDER = 1      # order of eddy-viscosity model
N_params_force = 0
eps = 30    # acceptance tolerance
########################################################################################################################
# Sample limits
C_limits = np.zeros((10, 2))
# C_limits[0] = [0.0, 0.4]
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
########################################################################################################################
# Parallel regime parameters
PROGRESSBAR = 1     # 0 - pool.map(no bar); 1 - pool.imap_unordered(progressbar); 2 - pool.map_async(text progress)
N_proc = 4          # Number of processes
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