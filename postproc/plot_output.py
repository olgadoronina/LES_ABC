import os
import yaml
import numpy as np
import plotting


# path_base = '../ABC/sigma_random/3_params_imcmc_random_100000_03domain/'
# path_base = '../ABC/sigma_random/4_params_imcmc_random_100000_03domain_N3400000/'
path_base = '../ABC/final/3_params_sigma/'
path = {'output': os.path.join(path_base, 'output'), 'visua': os.path.join(path_base, 'plots')}
if not os.path.isdir(path['visua']):
    os.makedirs(path['visua'])

params = yaml.load(open(os.path.join(path['output'], 'output_params.yml'), 'r'))

algorithm = params['abc']['algorithm']
########################
filename_calibration_all = os.path.join(path['output'], 'calibration_all.npz')
filename_calibration = os.path.join(path['output'], 'calibration.npz')
filename_accepted = os.path.join(path['output'], 'accepted_0.npz')

# if algorithm == 'IMCMC':
#     # Calibration
#     accepted = np.load(filename_calibration)['C']
#     dist = np.load(filename_calibration)['dist']
#     # plotting.plot_marginal_pdf(params['model']['N_params'], path['output'], path['visua'], params['C_limits'], '_calibration_')


accepted = np.load(filename_accepted)['C']
dist = np.load(filename_accepted)['dist']

C_limits = np.zeros((10, 2))
# C_limits[0] = [np.min(accepted), np.max(accepted)]
# C_limits[1] = [np.min(accepted), np.max(accepted)]
# C_limits[2] = [np.min(accepted), np.max(accepted)]
# C_limits[3] = [np.min(accepted), np.max(accepted)]
# C_limits[4] = [np.min(accepted), np.max(accepted)]
# C_limits[5] = [np.min(accepted), np.max(accepted)]

C_limits[0] = [np.min(accepted[:, 0]), 0.0]

# C_limits[0] = [np.min(accepted[:, 0]), np.max(accepted[:, 0])]
C_limits[1] = [np.min(accepted[:, 1]), np.max(accepted[:, 1])]
C_limits[2] = [np.min(accepted[:, 2]), np.max(accepted[:, 2])]
if params['model']['N_params'] == 4:
    C_limits[3] = [np.min(accepted[:, 3]), np.max(accepted[:, 3])]
# C_limits[4] = [np.min(accepted[:, 4]), np.max(accepted[:, 4])]
# C_limits[5] = [np.min(accepted[:, 5]), np.max(accepted[:, 5])]

print(C_limits)
# dist_calibration = np.load(filename_calibration_all)['S_init'][:, -1]
# plotting.plot_dist_pdf(dist_calibration, params['algorithm']['x'], path['visua'])
#
# plotting.plot_compare_tau(path['visua'], path['output'], params['compare_pdf']['summary_statistics'], scale='TEST')
# # plotting.plot_compare_tau(path['visua'], path['output'], params['compare_pdf']['summary_statistics'], scale='TEST_M')
# plotting.plot_marginal_pdf(params['model']['N_params'], path['output'], path['visua'], C_limits)

# C_limits[0] = [-0.3, 0.3]
# C_limits[1] = [-0.3, 0.3]
# C_limits[2] = [-0.3, 0.3]
plotting.plot_marginal_smooth_pdf_3(params['model']['N_params'], path['output'], path['visua'], C_limits)


# if algorithm == 'acc-rej':
#     new_eps = 3500
#     accepted = accepted[dist < new_eps]
#     dist = dist[dist < new_eps]
#     print('accepted {} values ({}%)'.format(len(accepted),
#                                                    round(len(accepted) / params['algorithm']['N_total'] * 100, 2)))

# plotting.plot_scatter(params['model']['N_params'], params['C_limits'], path['visua'], accepted, dist)
# plotting.plot_sum_stat(path, params['compare_pdf']['summary_statistics'])