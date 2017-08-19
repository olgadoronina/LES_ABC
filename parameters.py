from math import *

WORKDIR='./'
idir = WORKDIR+'/HIT_LES/test_N64/'  # input folder
odir = WORKDIR+'/HIT_LES/test_N64/'  # input folder
pid = 'test_LES_N64'                                # problem ID
L = 2.*pi                                           # domain size
N = 64                                              # linear grid size
cfl = 0.45                                          # CFL no.
tlimit = 10.*pi                                     # Time Limit
tau = 0.5*pi                                        # ~ eddy turnover time
dt_rst = 5.0*tau                                    # restart output rate
dt_bin = 1.0*tau                                    # snapshot output rate
dt_hst = 0.2*tau                                    # histogram output rate
dt_spec = 0.1*tau                                   # spectrum output rate
nu = 0.01                                           # kinematic viscosity
eps_inj = 0.6                               # energy injection rate (cell avg.)
Urms = 3.48                                 # initial rms velocity
k_exp = -1.0                                # initial spectrum power law
k_peak = 16                                 # initial spectrum decay scaling
