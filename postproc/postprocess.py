import os
import logging
import numpy as np

import abc_code.global_var as g
import abc_code.model as model
import abc_code.utils as utils
from params import output


########################################################################################################################
## class PostprocessABC
########################################################################################################################
class PostprocessABC(object):

    def __init__(self, C_limits, eps, num_bin_joint, params):

        logging.info('\nPostprocessing')
        assert len(g.accepted) != 0, 'Oops! No accepted values'
        self.num_bin_joint = num_bin_joint
        self.N_params = params.model['N_params']
        self.N_each = params.algorithm['N_each']
        self.model_params = params.model
        self.abc_algorithm = params.abc['algorithm']
        self.algorithm = params.algorithm
        self.bins = params.compare_pdf['bins']
        self.domain = params.compare_pdf['domain']
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
            logging.info('Min distance: {}'.format(g.dist[np.argmin(g.dist)]))
            np.savetxt(os.path.join(output['output_path'], 'C_final_dist'), self.C_final_dist)
        else:
            # C_final_dist
            minim = np.argmin(g.dist)
            self.C_final_dist = g.accepted[minim, :]
            C_final_dist = self.C_final_dist
            logging.info('Minimum distance is {} in: {}'.format(g.dist[minim], C_final_dist))
            np.savetxt(os.path.join(output['output_path'], 'C_final_dist'), self.C_final_dist)
            # C_final_joint
            H, edges = np.histogramdd(g.accepted, bins=self.num_bin_joint)
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
            np.savetxt(os.path.join(output['output_path'], 'C_final_joint'), self.C_final_joint)
            if len(ind) > 10:
                logging.warning('Can not estimate parameters from joint pdf!'
                                'Too many bins ({} bins, max value {}) '
                                'with the same max value in joint pdf'.format(len(ind), np.max(H)))
            else:
                logging.info('Estimated parameters from joint pdf: {}'.format(self.C_final_joint))

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
                        np.savetxt(os.path.join(output['output_path'], 'marginal'+str(i)), [x, y])
                    elif i < j:
                        H, xedges, yedges = np.histogram2d(x=g.accepted[:, j], y=g.accepted[:, i], bins=self.num_bin_joint)
                        np.savetxt(os.path.join(output['output_path'], 'marginal' + str(i)+str(j)), H)
                        np.savetxt(os.path.join(output['output_path'], 'marginal_bins' + str(i) + str(j)), [xedges, yedges])

    def calc_compare_sum_stat(self, scale='LES'):

        if self.N_params == 1:
            C_final_dist_new = self.C_final_dist[0].copy()
        else:
            C_final_dist_new = self.C_final_dist.copy()
        C_final_joint = 0
        if len(self.C_final_joint) < 4 and self.N_params != 1:
            # if len(self.C_final_joint) == 1:
            #     C_final_joint = self.C_final_joint[0].copy()
            #  else:
            C_final_joint = self.C_final_joint.copy()
        C_final_marginal = self.C_final_marginal

        # create pdfs
        if scale == 'LES':
            current_model = model.NonlinearModel(g.LES, self.model_params, self.abc_algorithm, self.algorithm, self.C_limits)
        if scale == 'TEST_M':
            current_model = model.NonlinearModel(g.TEST_sp, self.model_params, self.abc_algorithm, self.algorithm, self.C_limits)
        if scale == 'TEST':
            current_model = model.NonlinearModel(g.TEST, self.model_params, self.abc_algorithm, self.algorithm, self.C_limits)

        sigma_modeled_dist = current_model.sigma_from_C(C_final_dist_new)
        # sigma_modeled_marginal = current_model.sigma_from_C(C_final_marginal)

        for ind, key in enumerate(['uu', 'uv', 'uw']):
            # plot min dist pdf
            y = utils.pdf_from_array(sigma_modeled_dist[key].flatten(), self.bins, self.domain)
            y = utils.take_safe_log(y)
            np.savetxt(os.path.join(output['output_path'], 'sum_stat_min_dist'), y)
            # # plot max marginal
            # y = utils.pdf_from_array(sigma_modeled_marginal[key].flatten(), self.bins, self.domain)
            # y = utils.take_safe_log(y)
            # np.savetxt(os.path.join(output['output_path'], 'sum_stat_max_marginal'), y)
        # Plot max joint pdf
        if C_final_joint:
            for i in range(len(C_final_joint)):
                sigma_modeled_joint = current_model.sigma_from_C(C_final_joint[i])
                y_dict = dict()
                for ind, key in enumerate(['uu', 'uv', 'uw']):
                    y_dict[key] = utils.pdf_from_array(sigma_modeled_joint[key].flatten(), self.bins, self.domain)
                    y = utils.take_safe_log(y_dict[key])
                    np.savetxt(os.path.join(output['output_path'], 'sum_stat_min_dist'), y)





