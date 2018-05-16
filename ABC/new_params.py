from math import pi

params = {
    ########################################################################################################################
    # Data
    'data': {'load': True,
            'data_name': 'JHU_data',
            'data_path': './data_input/JohnHopkins/'},
    # 'data_name': 'CU_data'
    # 'data_folder':  './data_input/HIT_DNS_N256/'}
    'output': {'output_path': './plots/'},
    ########################################################################################################################
    # Initial case parameters
    'physical_case': {'N_point': 256,
                     'N_points': [256, 256, 256],      # number of points
                     'lx': [2 * pi, 2 * pi, 2 * pi],               # domain size
                     'LES_scale': 30/2/pi,
                     'TEST_scale':None},
    # Filter scales
    # LES_scale = 10
    # TEST_scale = 5
    ########################################################################################################################
    # Model parameters
    'model': {'HOMOGENEOUS': 1,     # Use symmetry of tau tensor
             'ORDER': 3,           # order of eddy-viscosity model
             'N_params_force': 6},

    ########################################################################################################################
    # Sampling
    'sampling': {'sampling': 'uniform',    # 'uniform', 'random', 'sobol' , 'MCMC'
                'N_each':100,
                'N_params_in_task':2},  # only 0, 1 or 2  #only 0 and 2 for calibration

    'compare_pdf': {'bins': 100,  # for pdf comparison
                   'domain': [-0.45, 0.45],  # for pdf comparison
                   'distance': 'L2log'},
    # domain = [-0.7, 0.7]  # for pdf comparison
    #                'num_bin_joint': 20
    ########################################################################################################################
    # abs algorithm
    'abc_algorithm': {'algorithm': 1,    # 0 = uniform sampling; 1 = MCMC; 2 = IMCMC; 3 = Gaussian mixture; 4 = PMC
                         {'eps': 2000,  # acceptance tolerance
                          }},
    ########################################################################################################################
    'calibration': {'N_calibration': 0,  # recommended 10^p, where p is number of params
                   'x': 0.01,   # percent of accepted for calibration step
                   'phi': 1},
    ########################################################################################################################
    'MCMC': {'algorithm': 1,    # 0 = uniform sampling; 1 = MCMC; 2 = IMCMC; 3 = Gaussian mixture; 4 = PMC
            'N_total': 10**7},


    # M = 64                                     # number of training points (sparse data)
    # N_gaussians = 30  # number of gaussians
    ########################################################################################################################

    #######################################################################################################################
    'C_limits': {0: [-0.3, 0.3],
                1: [-0.5, 0.5],
                2: [-0.2, 0.2],
                3: [-0.2, 0.2],
                4: [-1, 1],
                5: [-0.5, 0.5],
                6: [-1, 1],
                7: [-1, 1],
                8: [-1, 1],
                9: [-1, 1]},

    ################################
    # var = np.empty(10)
    # var = (C_limits[:, 1] - C_limits[:, 0])/2
    ########################################################################################################################
    # Parallel regime parameters
    'parallel': {'PROGRESSBAR': 1,    # 0 - pool.map(no bar); 1 - pool.imap_unordered(progressbar); 2 - pool.map_async(text progress)
                'N_proc': 6}          # Number of processes
    ########################################################################################################################
}
