from math import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from time import time, sleep
import gc
import random as rand
import logging
# import cProfile
########################################################################################################################
# mpl.style.use(['dark_background','mystyle'])
# mpl.style.use(['mystyle'])
plt.style.use('seaborn-white')
plt.rc('font', **{'family':'serif', 'serif':['Computer Modern Roman']})
plt.rc('text', usetex=True)
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
# plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 18
# plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.titlesize'] = 22
########################################################################################################################
logging.basicConfig(format='%(levelname)s: %(name)s: %(message)s', level=logging.DEBUG)
# logging.basicConfig(filename='ABC_log.log', filemode='w',
#                     format='%(levelname)s: %(name)s: %(message)s',
#                     level=logging.DEBUG)
# prof = cProfile.Profile()
########################################################################################################################
TINY = 1e-09
TINY_log = np.log(TINY)
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
# LES_scale = 10
# TEST_scale = 5
LES_scale = 30/2/pi
TEST_scale = 15/2/pi
# Characteristic length \Delta
LES_delta = 1/LES_scale
TEST_delta = 1/TEST_scale
########################################################################################################################
# abs algorithm
bins = 100  # for pdf
num_bin_joint = 20
N_each = 1000
N_params = 1
N_params_in_task = 0  # only 0, 1 or 2

N_total = N_each**N_params
# step = int(65000/2) #floor((32+256)*1024/8/N_each)-100  # 32KB L1 cache, 256KB L2 cache
# logging.info('step = ', step)

M = 64          # number of training points
ORDER = 1      # order of eddy-viscosity model
USE_C3 = 0
eps = 170     # acceptance tolerance
########################################################################################################################
# Params for abc algorithm
domain = [-1.1, 1.1]  # for pdf comparison
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
