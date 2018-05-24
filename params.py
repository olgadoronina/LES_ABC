from math import pi

########################################################################################################################
# Data
data = {'load': True,
        'data_name': 'JHU_data',
        'data_path': './data_input/JohnHopkins/'}
# 'data_name': 'CU_data'
# 'data_folder':  './data_input/HIT_DNS_N256/'}
path = {'output': './output/',
        'visua': './plots/'}
########################################################################################################################
# Initial case parameters
physical_case = {'N_point': 256,
                 'lx': 2 * pi,          # domain size
                 'homogeneous': True,   # Use symmetry of tau tensor
                 'LES_scale': 30/2/pi,
                 'TEST_scale': None}
                 # LES_scale = 10
                 # TEST_scale = 5
########################################################################################################################
# Model parameters
model = {'order': 2,           # order of eddy-viscosity model
         'N_params_force': 3}
########################################################################################################################
compare_pdf = {'bins': 100,                 # for pdf comparison
               'domain': [-0.45, 0.45],     # for pdf comparison
               # domain = [-0.7, 0.7]      # for pdf comparison
               'distance': 'L2log',
               'summary_statistics': 'sigma_pdf_log'}  # 'sigma_pdf_log', 'production_pdf_log'; production mean
########################################################################################################################
# abs algorithm
abc = {'algorithm': 'IMCMC',    # 'acc-rej' = acceptance-rejection; 'MCMC'; 'IMCMC'; 'AGM-MH'= Gaussian mixture; 'PMC'
       'num_training_points': 64}
################################################################################################################
# Define only one of the following
################################################################################################################
algorithm ={'acc-rej':
               {'sampling': 'uniform',   # 'uniform', 'random', 'sobol'
                'eps': 5000,
                'N_each': 10,
                'N_params_in_task': 0},  # only 0, 1 or 2  #only 0 and 2 for calibration
            ############################################################################################################
            'MCMC':
               {'N_total_chain': 1e7,
                'eps': 5000,
                'var_multiplier': 0.5},
            ############################################################################################################
            'IMCMC':
                # Calibration
               {'sampling': 'uniform',     # 'uniform', 'random', 'sobol'
                'N_each': 10,              # recommended 10
                'N_params_in_task': 0,     # only 0 and 2 for calibration
                'x': 0.15,                 # percent of accepted for calibration step
                'phi': 1,
                # MCMC
                'N_total_chain': 1e7},
            ############################################################################################################
            'AGM-MH':
                {' N_gaussians': 30,
                 'var_multiplier': 0.5,
                 'eps': 2000}
            }
########################################################################################################################

#######################################################################################################################
C_limits = [[-0.3, 0.3],
            [-0.5, 0.5],
            [-0.2, 0.2],
            [-0.2, 0.2],
            [-1, 1],
            [-0.5, 0.5],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1]]
########################################################################################################################
# Parallel regime parameters
parallel = {'progressbar': 1,    # 0 - pool.map(no bar); 1 - pool.imap_unordered(progressbar); 2 - pool.map_async(text progress)
            'N_proc': 4}          # Number of processes
########################################################################################################################

