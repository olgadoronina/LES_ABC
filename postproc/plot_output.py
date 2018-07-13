import os
import yaml
import numpy as np
import plotting
import logging

path_base = '../ABC/3_params_sigma_imcmc_summit/'
path = {'output': os.path.join(path_base, 'output'),
        'visua': os.path.join(path_base, 'plots')}
if not os.path.isdir(path['visua']):
    os.makedirs(path['visua'])

params = yaml.load(open(os.path.join(path['output'], 'output_params.yml'), 'r'))

########################
calibration = 0
filename_calibration_all = os.path.join(path['output'], 'calibration_all.npz')
filename_calibration = os.path.join(path['output'], 'calibration.npz')
filename_accepted = os.path.join(path['output'], 'accepted.npz')
if calibration:
    filename = filename_calibration
else:
    filename = filename_accepted
accepted = np.load(filename)['C']
dist = np.load(filename)['dist']



# plotting.plot_scatter(params['model']['N_params'], params['C_limits'], path['visua'], accepted, dist)
########################
if not calibration:
    plotting.plot_compare_tau(path['visua'], path['output'], params['compare_pdf']['summary_statistics'], scale='TEST')
    plotting.plot_compare_tau(path['visua'], path['output'], params['compare_pdf']['summary_statistics'], scale='TEST_M')

plotting.plot_marginal_pdf(params['model']['N_params'], path['output'], path['visua'], params['C_limits'])

# plotting.plot_sum_stat(path, params['compare_pdf']['summary_statistics'])