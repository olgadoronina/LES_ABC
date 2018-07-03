import os
import yaml
import numpy as np
import plotting
import logging

path_base = '../ABC/'
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

new_eps = 3000
accepted = accepted[dist < new_eps]
dist = dist[dist < new_eps]
logging.info('accepted {} values ({}%)'.format(len(accepted),
                                               round(len(accepted) / params['algorithm']['N_total'] * 100, 2)))

plotting.plot_scatter(params['model']['N_params'], params['C_limits'], path['visua'], accepted, dist)
########################
# plotting.plot_compare_tau(path['visua'], path['output'], params['compare_pdf']['summary_statistics'], scale='TEST')
plotting.plot_compare_tau(path['visua'], path['output'], params['compare_pdf']['summary_statistics'], scale='TEST_M')

# plotting.plot_sum_stat(path, params['compare_pdf']['summary_statistics'])