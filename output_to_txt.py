import logging
import abc_code.data as data
import abc_code.global_var as g
import numpy as np
# import postproc.plotting as plotting
import init
import os
import sys
import yaml
import abc_code.model as model
import abc_code.utils as utils
########################################################################################################################
## class PostprocessABC
########################################################################################################################
class PostprocessABC(object):

    def __init__(self, C_limits, eps, num_bin_joint, params):

        logging.info('\nPostprocessing')
        assert len(g.accepted) != 0, 'Oops! No accepted values'
        self.num_bin_joint = num_bin_joint
        self.N_params = params['model']['N_params']
        self.N_each = params['algorithm']['N_each']
        self.model_params = params['model']
        self.abc_algorithm = params['abc']['algorithm']
        self.algorithm = params['algorithm']
        self.pdf_params = params['compare_pdf']
        self.bins = params['compare_pdf']['bins']
        self.domain = params['compare_pdf']['domain']
        if self.N_params != len(g.accepted[0]):
            self.N_params = len(g.accepted[0])
            logging.warning('Wrong number of params in params.py. Use {} params'.format(self.N_params))
        self.C_limits = C_limits
        self.eps = eps

        self.params_names = [r'$C_1$', r'$C_2$', r'$C_3$', r'$C_4$', r'$C_5$', r'$C_6$', r'$C_7$', r'$C_8$', r'$C_9$']
        self.C_final_dist = []
        self.C_final_joint = []
        self.C_final_marginal = np.empty(self.N_params)

    def calc_final_C(self):
        """ Estimate the best fit of parameters.
        For 1 parameter: based on minimum distance;
        For more then 1 parameter: based on joint pdf.
        """

        if self.N_params == 1:
            # C_final_dist only
            self.C_final_dist = [[g.accepted[:, 0][np.argmin(g.dist)]]]
            logging.info('Estimated parameter: {}'.format(self.C_final_dist[0][0]))
            min_dist = g.dist[np.argmin(g.dist)]
            logging.info('Min distance: {}'.format(min_dist))
            np.savetxt(os.path.join(path['output'], 'C_final_dist'), self.C_final_dist)
            f = open(os.path.join(path['output'], 'C_final_dist'), 'a')
            np.savetxt(f, np.array([min_dist]))
            f.close()
        else:
            # C_final_dist
            minim = np.argmin(g.dist)
            self.C_final_dist = g.accepted[minim, :]
            C_final_dist = self.C_final_dist
            min_dist = g.dist[minim]
            logging.info('Minimum distance is {} in: {}'.format(g.dist[minim], C_final_dist))
            np.savetxt(os.path.join(path['output'], 'C_final_dist'), self.C_final_dist)
            np.savetxt(os.path.join(path['output'], 'C_final_dist'), min_dist)
            # C_final_joint
            H, edges = np.histogramdd(g.accepted, bins=self.num_bin_joint, range=self.C_limits)
            logging.debug('Max number in bin: {}'.format(np.max(H)))
            logging.debug('Mean number in bin: {}'.format(np.mean(H)))
            edges = np.array(edges)
            C_bin = (edges[:, :-1] + edges[:, 1:]) / 2  # shift value in the center of the bin
            ind = np.argwhere(H == np.max(H))
            for i in ind:
                point = []
                for j in range(self.N_params):
                    point.append(C_bin[j, i[j]])
                self.C_final_joint.append(point)
            np.savetxt(os.path.join(path['output'], 'C_final_joint'), self.C_final_joint)
            if len(ind) > 10:
                logging.warning('Can not estimate parameters from joint pdf!'
                                'Too many bins ({} bins, max value {}) '
                                'with the same max value in joint pdf'.format(len(ind), np.max(H)))
            else:
                logging.info('Estimated parameters from joint pdf: {}'.format(self.C_final_joint))

########################################################################################################################
    def calc_marginal_pdf(self):

        if self.N_params != 1:
            for i in range(self.N_params):
                for j in range(self.N_params):
                    if i == j:
                        # mean = np.mean(g.accepted[:, i])
                        x, y = utils.pdf_from_array_with_x(g.accepted[:, i], bins=self.N_each, range=self.C_limits[i])
                        # max = x[np.argmax(y)]
                        # print('{} marginal mean is {} and max is {}'. format(self.params_names[i], mean, max))
                        # self.C_final_marginal[i] = max
                        np.savetxt(os.path.join(path['output'], 'marginal'+str(i)), [x, y])
                    elif i < j:
                        H, xedges, yedges = np.histogram2d(x=g.accepted[:, j], y=g.accepted[:, i],
                                                           bins=self.num_bin_joint,
                                                           range=[self.C_limits[j], self.C_limits[i]])
                        np.savetxt(os.path.join(path['output'], 'marginal' + str(i)+str(j)), H)
                        np.savetxt(os.path.join(path['output'], 'marginal_bins' + str(i) + str(j)), [xedges, yedges])

########################################################################################################################
    def calc_compare_sum_stat(self, sum_stat, scale='LES'):

        if self.N_params == 1:
            C_final_dist_new = self.C_final_dist[0].copy()
        else:
            C_final_dist_new = self.C_final_dist.copy()
        C_final_joint = 0
        if len(self.C_final_joint) < 4 and self.N_params != 1:
            C_final_joint = self.C_final_joint.copy()
        C_final_marginal = self.C_final_marginal

        # create pdfs
        if scale == 'LES':
            current_model = model.NonlinearModel(g.LES, self.model_params, self.abc_algorithm, self.algorithm,
                                                 self.C_limits, self.pdf_params)
        if scale == 'TEST_M':
            current_model = model.NonlinearModel(g.TEST_sp, self.model_params, self.abc_algorithm, self.algorithm,
                                                 self.C_limits,  self.pdf_params)
        if scale == 'TEST':
            current_model = model.NonlinearModel(g.TEST, self.model_params, self.abc_algorithm, self.algorithm,
                                                 self.C_limits,  self.pdf_params)

        sigma_modeled_dist = current_model.sigma_from_C(C_final_dist_new)

        # sigma_modeled_marginal = current_model.sigma_from_C(C_final_marginal)

        # calc min dist pdf
        if sum_stat == 'sigma_pdf_log':
            y = np.empty((3, self.bins))
            for ind in range(3):
                y[ind] = utils.take_safe_log(sigma_modeled_dist[ind])
        elif sum_stat == 'production_pdf_log':
            y = utils.take_safe_log(sigma_modeled_dist)
        np.savetxt(os.path.join(path['output'], 'sum_stat_min_dist_' + scale), y)
            # # plot max marginal
            # y = utils.pdf_from_array(sigma_modeled_marginal[key].flatten(), self.bins, self.domain)
            # y = utils.take_safe_log(y)
            # np.savetxt(os.path.join(output['output_path'], 'sum_stat_max_marginal'), y)

        # calc max joint pdf
        if C_final_joint:
            for i in range(len(C_final_joint)):
                sigma_modeled_joint = current_model.sigma_from_C(C_final_joint[i])
                if sum_stat == 'sigma_pdf_log':
                    for ind in range(3):
                        tmp = utils.pdf_from_array(sigma_modeled_joint[ind], self.bins, self.domain)
                        y = utils.take_safe_log(tmp)
                elif sum_stat == 'production_pdf_log':
                    tmp = utils.pdf_from_array(sigma_modeled_joint, self.bins, self.domain)
                    y = utils.take_safe_log(tmp)
                np.savetxt(os.path.join(path['output'], 'sum_stat_max_joint_' + scale), y)




path_base = './ABC/3_params_sigma_uniform/'
path = {'output': os.path.join(path_base, 'output'),
        'visua': os.path.join(path_base, 'plots')}
# if not os.path.isdir(path['visua']):
#     os.makedirs(path['visua'])

uniform = 1
calibration = 0
IMCMC = 0

filename_calibration_all = os.path.join(path['output'], 'calibration_all.npz')
filename_calibration = os.path.join(path['output'], 'calibration.npz')
filename_accepted = os.path.join(path['output'], 'accepted.npz')
if calibration:
    filename = filename_calibration
else:
    filename = filename_accepted

print(filename)

logging.basicConfig(format='%(levelname)s: %(name)s: %(message)s', level=logging.DEBUG)
# ####################################################################################################################
# # Initial data
# ####################################################################################################################
params = yaml.load(open(os.path.join(path['output'], 'output_params.yml'), 'r'))
g.path = params['path']
params['data']['data_path'] = os.path.join(path_base, 'data_input/'+params['data']['data_name'])
print(params)
init.LES_TEST_data(params['data'], params['physical_case'], params['compare_pdf'])
g.TEST_sp = data.DataSparse(g.TEST, params['abc']['num_training_points'])


########################
g.accepted = np.load(filename)['C']
g.dist = np.load(filename)['dist']
########################

if calibration:
    C_limits = params['C_limits']
    num_bin_joint = 10
    N_each = 10
    dist = np.load(filename_calibration_all)['S_init'][:, -1]
    # plotting.dist_pdf(dist, params['algorithm']['x'], params['path']['visua'])

else:
    num_bin_joint = 20
    N_each = 100
    C_limits = params['C_limits']
    # C_limits = np.zeros((10, 2))
    # C_limits[0] = [np.min(g.accepted[:, 0]), np.max(g.accepted[:, 0])]
    # C_limits[1] = [np.min(g.accepted[:, 1]), np.max(g.accepted[:, 1])]
    # C_limits[2] = [np.min(g.accepted[:, 2]), np.max(g.accepted[:, 2])]
    # C_limits[3] = [np.min(g.accepted[:, 3]), np.max(g.accepted[:, 3])]
    # C_limits[4] = [np.min(g.accepted[:, 4]), np.max(g.accepted[:, 4])]
    # C_limits[5] = [np.min(g.accepted[:, 5]), np.max(g.accepted[:, 5])]
# # # #########################

eps = g.eps
params['algorithm']['N_each'] = N_each
postproc = PostprocessABC(C_limits, eps, num_bin_joint, params)
#
#
# if uniform:
#     new_eps = 10
#     g.accepted = g.accepted[g.dist < new_eps]
#     g.dist = g.dist[g.dist < new_eps]
#     logging.info('accepted {} values ({}%)'.format(len(g.accepted),
#                                                    round(len(g.accepted) / params['algorithm']['N_total'] * 100, 2)))
#
#
postproc.calc_final_C()
# postproc.calc_marginal_pdf()


if not calibration:

    # postproc.plot_eps()

    postproc.calc_compare_sum_stat(params['compare_pdf']['summary_statistics'], scale='TEST')

    postproc.calc_compare_sum_stat(params['compare_pdf']['summary_statistics'], scale='TEST_M')
