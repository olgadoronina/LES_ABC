import logging
import abc_code.data as data
import abc_code.global_var as g
import postproc.kde as kde

import numpy as np
import postproc.plotting as plotting
import init
import os
import sys
import yaml
import abc_code.model as model
import abc_code.utils as utils
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
logging.basicConfig(format='%(levelname)s: %(name)s: %(message)s', level=logging.DEBUG)
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
        self.N_total = params['algorithm']['N_total']
        self.model_params = params['model']
        self.abc = params['abc']
        self.algorithm = params['algorithm']
        self.pdf_params = params['compare_pdf']
        self.bins = params['compare_pdf']['bins']
        self.domain = params['compare_pdf']['domain']
        if self.N_params != len(g.accepted[0]):
            self.N_params = len(g.accepted[0])
            logging.warning('Wrong number of params in params.py. Use {} params'.format(self.N_params))
        self.C_limits = C_limits
        self.C_limits_init = params['C_limits']
        self.eps = eps

        self.params_names = [r'$C_1$', r'$C_2$', r'$C_3$', r'$C_4$', r'$C_5$', r'$C_6$', r'$C_7$', r'$C_8$', r'$C_9$']
        self.C_final_dist = []
        self.C_final_joint = []
        self.C_final_smooth = []
        self.C_final_marginal = np.empty(self.N_params)
        self.Z = []
        self.ind_max = []

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
            # min_dist = g.dist[minim]
            logging.info('Minimum distance is {} in: {}'.format(g.dist[minim], C_final_dist))
            np.savetxt(os.path.join(path['output'], 'C_final_dist'), self.C_final_dist)
            # np.savetxt(os.path.join(path['output'], 'C_final_dist'), min_dist)

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
            np.savetxt(os.path.join(path['output'], 'C_final_joint'+str(self.num_bin_joint)), self.C_final_joint)
            if len(ind) > 10:
                logging.warning('Can not estimate parameters from joint pdf!'
                                'Too many bins ({} bins, max value {}) '
                                'with the same max value in joint pdf'.format(len(ind), np.max(H)))
            else:
                logging.info('Estimated parameters from joint pdf: {}'.format(self.C_final_joint))
            #
            # # Gaussian smoothness
            self.Z, self.C_final_smooth, self.ind_max = kde.gaussian_kde_scipy(g.accepted, self.C_limits[:self.N_params, 0],
                                                                 self.C_limits[:self.N_params, 1], self.num_bin_joint)



            # np.savetxt(os.path.join(path['output'], 'C_final_smooth'+str(self.num_bin_joint)), self.C_final_smooth)
            np.savetxt(os.path.join(path['output'], 'C_final_smooth'), self.C_final_smooth)
            # np.savetxt(os.path.join(path['output'], 'posterior' + str(self.num_bin_joint)), Z)
            logging.info('Estimated parameters from joint pdf: {}'.format(self.C_final_smooth))
    ####################################################################################################################

    def calc_marginal_pdf(self, name=''):

        if self.N_params != 1:
            for i in range(self.N_params):
                for j in range(self.N_params):
                    if i == j:
                        x, y = utils.pdf_from_array_with_x(g.accepted[:, i], bins=self.N_each, range=self.C_limits_init[i])
                        # mean = np.mean(g.accepted[:, i])
                        # max = x[np.argmax(y)]
                        # print('{} marginal mean is {} and max is {}'. format(self.params_names[i], mean, max))
                        # self.C_final_marginal[i] = max
                        # np.savetxt(os.path.join(path['output'], 'marginal' + name + str(i)), [x, y])
                        # Smooth
                        y = np.sum(self.Z, axis=tuple(np.where(np.arange(self.N_params) != i)[0]))
                        x = np.linspace(self.C_limits[i, 0], self.C_limits[i, 1], self.num_bin_joint + 1)
                        np.savetxt(os.path.join(path['output'], 'marginal_smooth' + name + str(i)), [x, y])
                    elif i < j:

                        H, xedges, yedges = np.histogram2d(x=g.accepted[:, j], y=g.accepted[:, i],
                                                           bins=self.num_bin_joint,
                                                           range=[self.C_limits_init[j], self.C_limits_init[i]])
                        np.savetxt(os.path.join(path['output'], 'marginal' + name + str(i)+str(j)), H)
                        np.savetxt(os.path.join(path['output'], 'marginal_bins' + name + str(i) + str(j)),
                                   [xedges, yedges])

                        # Smooth
                        params = np.arange(self.N_params)
                        ind = tuple(np.where(np.logical_and(params != i, params != j))[0])
                        H = np.sum(self.Z, axis=ind)

                        np.savetxt(os.path.join(path['output'], 'marginal_smooth' + name + str(i) + str(j)), H)

########################################################################################################################
    def calc_conditional_pdf(self, name=''):

        if self.N_params != 1:
            for i in range(self.N_params):
                for j in range(self.N_params):
                    if i > j:
                        # Smooth
                        params = np.arange(self.N_params)
                        ind = tuple(np.where(np.logical_and(params != i, params != j))[0])
                        if self.N_params == 3:
                            H = np.take(self.Z, self.ind_max[ind], axis=ind[0])
                        np.savetxt(os.path.join(path['output'], 'conditional_smooth' + name + str(i) + str(j)), H)

    ########################################################################################################################
    def calc_compare_sum_stat(self, output, sum_stat, scale='LES'):

        # if self.N_params == 1:
        #     C_final_dist = self.C_final_dist[0].copy()
        # else:
        #     C_final_dist = self.C_final_dist.copy()
        # C_final_joint = 0
        # if len(self.C_final_joint) < 4 and self.N_params != 1:
        #     C_final_joint = self.C_final_joint.copy()


        # C_final_marginal = self.C_final_marginal

        # create model
        if scale == 'LES':
            current_model = model.NonlinearModel(g.path['data_path'], 0, self.model_params, self.abc_algorithm,
                                                 self.algorithm, self.C_limits,  self.pdf_params, 0, g.LES)
        if scale == 'TEST_M':
            current_model = model.NonlinearModel(g.path['data_path'], 0, self.model_params, self.abc,
                                                 self.algorithm, self.C_limits,  self.pdf_params, 0, g.TEST_sp)
        if scale == 'TEST':
            current_model = model.NonlinearModel(g.path['data_path'], 0, self.model_params, self.abc,
                                                 self.algorithm, self.C_limits,  self.pdf_params, 0, g.TEST)

        # sigma_modeled_dist = current_model.sigma_from_C(C_final_dist)
        # sigma_modeled_marginal = current_model.sigma_from_C(C_final_marginal)

        # calc min dist pdf
        # if sum_stat == 'sigma_pdf_log':
        #     y = np.empty((3, self.bins))
        #     for ind in range(3):
        #         y[ind] = utils.take_safe_log(sigma_modeled_dist[ind])
        # elif sum_stat == 'production_pdf_log':
        #     y = utils.take_safe_log(sigma_modeled_dist)
        # np.savetxt(os.path.join(path['output'], 'sum_stat_min_dist_' + scale), y)
        #     # # plot max marginal
            # y = utils.pdf_from_array(sigma_modeled_marginal[key].flatten(), self.bins, self.domain)
            # y = utils.take_safe_log(y)
            # np.savetxt(os.path.join(output['output_path'], 'sum_stat_max_marginal'), y)

        # # calc max joint pdf
        # if C_final_joint:
        #     for i in range(len(C_final_joint)):
        #         sigma_modeled_joint = current_model.sigma_from_C(C_final_joint[i])
        #
        #         if sum_stat == 'sigma_pdf_log':
        #             y = np.empty((4, self.bins))
        #             for ind in range(3):
        #                 y[ind] = utils.take_safe_log(sigma_modeled_joint[ind])
        #         elif sum_stat == 'production_pdf_log':
        #             y = utils.take_safe_log(sigma_modeled_joint[ind])
        #         np.savetxt(os.path.join(path['output'], 'sum_stat_max_joint_' + scale), y)

        # calc max joint smooth pdf
        C_final_smooth = [np.loadtxt(os.path.join(output, 'C_final_smooth'))]

        for i in range(len(C_final_smooth)):
            sigma_modeled_smooth = current_model.sigma_pdf(C_final_smooth[i])
            y = np.empty((4, self.bins))
            for ind in range(3):
                y[ind] = utils.take_safe_log(sigma_modeled_smooth[ind])
                production_modeled_joint = current_model.production_pdf(C_final_smooth[i])
                y[3] = utils.take_safe_log(production_modeled_joint)
        np.savetxt(os.path.join(path['output'], 'sum_stat_max_smooth_' + scale), y)

    def plot_eps(self):

        num_eps = 15
        eps = np.linspace(500, 5000, num_eps)

        C_mean = np.empty((self.N_params, num_eps))
        C_max = np.empty_like(C_mean)
        C_std = np.empty((self.N_params, num_eps))
        C_h = np.empty((self.N_params, num_eps))
        percent_accepted = np.empty(num_eps)
        max_joint = []
        min_dist = np.empty((self.N_params, num_eps))

        fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(6.5, 2.5))
        for ind, new_eps in enumerate(eps):
            accepted = g.accepted.copy()
            dist = g.dist.copy()

            accepted = accepted[dist < new_eps]
            dist = g.dist[dist < new_eps]
            logging.info('eps = {}: accepted {} values ({}%)'.format(new_eps,
                len(accepted), round(len(accepted) / (self.N_total) * 100, 2)))
            percent_accepted[ind] = len(accepted) / (self.N_total) * 100

            minim = np.argmin(dist)
            C_final_dist = accepted[minim, :]
            min_dist[:,ind] = C_final_dist
            # C_final_joint
            H, edges = np.histogramdd(g.accepted, bins=self.num_bin_joint, range=self.C_limits)
            edges = np.array(edges)
            C_bin = (edges[:, :-1] + edges[:, 1:]) / 2  # shift value in the center of the bin
            index = np.argwhere(H == np.max(H))
            C_final_joint = []
            for i in index:
                point = []
                for j in range(self.N_params):
                    point.append(C_bin[j, i[j]])
                C_final_joint.append(point)
            max_joint.append(C_final_joint)

            for i in range(self.N_params):
                data = accepted[:, i]
                C_std[i, ind] = np.std(data)
                C_mean[i, ind], C_h[i, ind] = utils.mean_confidence_interval(data, confidence=0.95)
                x, y = utils.pdf_from_array_with_x(data, bins=self.N_each, range=self.C_limits[i])
                C_max[i, ind] = x[np.argmax(y)]
                axarr[i].plot(x, y, label=r'$\epsilon \approx {}$'.format(np.round(new_eps)))
                axarr[i].set_xlabel(self.params_names[i])

        axarr[0].set_ylabel('marginal pdf')
        plt.legend(loc='upper center', bbox_to_anchor=(1., 1.1))
        # frame = legend.get_frame()
        # frame.set_alpha(1)
        fig.subplots_adjust(left=0.15, right=0.9, wspace=0.17, bottom=0.17, top=0.9)
        fig.savefig(os.path.join(path['visua'], 'eps_marginal'))

        fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(6.5, 2.5))
        for i in range(self.N_params):
            axarr[i].plot(eps, C_mean[i], 'b.-', label='mean')
            axarr[i].plot(eps, C_max[i], 'g.-', label='max')
            axarr[i].set_title(self.params_names[i])
            axarr[i].set_xlabel('epsilon')
            # axarr[i].xaxis.set_major_locator(ticker.MultipleLocator(5))
        axarr[0].set_ylabel(r'$C_i$')
        plt.legend(loc=0)
        fig.subplots_adjust(left=0.1, right=0.97, wspace=0.4, bottom=0.2, top=0.85)
        fig.savefig(os.path.join(path['visua'],'eps_plot'))


        fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(6.5, 2.5))
        for i in range(self.N_params):
            axarr[i].plot(eps, C_std[i], 'b.-')
            axarr[i].set_title(self.params_names[i])
            axarr[i].set_xlabel('epsilon')
            # axarr[i].xaxis.set_major_locator(ticker.MultipleLocator(5))
            # axarr[i].axis(ymin=np.min(C_mean[i])-0.01*, ymax=np.max(C_mean[i])+0.1)
        axarr[0].set_ylabel(r'std($C_i$)')
        fig.subplots_adjust(left=0.15, right=0.97, wspace=0.4, bottom=0.2, top=0.85)
        fig.savefig(os.path.join(path['visua'], 'eps_std'))


        fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(6.5, 2.5))
        for i in range(self.N_params):
            axarr[i].plot(eps, C_h[i], 'b.-')
            axarr[i].set_title(self.params_names[i])
            axarr[i].set_xlabel('epsilon')
            # axarr[i].xaxis.set_major_locator(ticker.MultipleLocator(5))
            # axarr[i].axis(ymin=np.min(C_mean[i])-0.01*, ymax=np.max(C_mean[i])+0.1)
        axarr[0].set_ylabel(r'$95\%$ confident interval')
        fig.subplots_adjust(left=0.12, right=0.97, wspace=0.4, bottom=0.2, top=0.85)
        fig.savefig(os.path.join(path['visua'], 'eps_h'))

        fig = plt.figure(figsize=(4, 3))
        ax = plt.gca()
        ax.plot(eps, percent_accepted, 'b.-')
        ax.set_xlabel('epsilon')
            # axarr[i].xaxis.set_major_locator(ticker.MultipleLocator(5))
            # axarr[i].axis(ymin=np.min(C_mean[i])-0.01*, ymax=np.max(C_mean[i])+0.1)
        ax.set_ylabel('accepted samples in percents [%]')
        fig.subplots_adjust(left=0.12, right=0.97, bottom=0.2, top=0.85)
        fig.savefig(os.path.join(path['visua'], 'percent_accepted'))

        # fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(6.5, 2.5))
        # for i in range(self.N_params):
        #     axarr[i].plot(eps, min_dist[i], 'g.')
        #     axarr[i].set_title(self.params_names[i])
        #     axarr[i].set_xlabel('epsilon')
        #     for ind in range(num_eps):
        #         print(max_joint[ind])
        #         for j in max_joint[ind]:
        #             axarr[i].scatter(eps[ind], j[i], color='b')
        #     # axarr[i].xaxis.set_major_locator(ticker.MultipleLocator(5))
        #     # axarr[i].axis(ymin=np.min(C_mean[i])-0.01*, ymax=np.max(C_mean[i])+0.1)
        # axarr[0].set_ylabel(r'$C_i$')
        # fig.subplots_adjust(left=0.12, right=0.97, wspace=0.4, bottom=0.2, top=0.85)
        # fig.savefig(os.path.join(path['visua'], 'eps_parameter'))


# ####################################################################################################################
# # Script starts here
# ####################################################################################################################
path_base = '../ABC/final/3_params_sigma/'
path = {'output': os.path.join(path_base, 'output'), 'visua': os.path.join(path_base, 'plots')}
if not os.path.isdir(path['visua']):
    os.makedirs(path['visua'])
params = yaml.load(open(os.path.join(path['output'], 'output_params.yml'), 'r'))
g.path = path
params['data']['data_path'] = os.path.join('../ABC/data_input/'+params['data']['data_name'])
g.path['data_path'] = params['data']['data_path']
print(params)

algorithm = params['abc']['algorithm']

filename_calibration_all = os.path.join(path['output'], 'calibration_all.npz')
filename_calibration = os.path.join(path['output'], 'calibration.npz')
filename_accepted = os.path.join(path['output'], 'accepted_0.npz')

# ####################################################################################################################
# # Initial data
# ####################################################################################################################

init.load_LES_TEST_data(params['data'], params['physical_case'], params['compare_pdf'])
if params['abc']['random'] == 0:
    g.TEST_sp = data.DataSparse(params['data']['data_path'], 0, g.TEST, params['abc']['num_training_points'])
#######################
# if algorithm == 'IMCMC':
#     # Calibration
#     g.accepted = np.load(filename_calibration)['C']
#     g.dist = np.load(filename_calibration)['dist']
#
#     C_limits = params['C_limits']
#     num_bin_joint = 10
#     N_each = 10
#     dist = np.load(filename_calibration_all)['S_init'][:, -1]
#     # plotting.dist_pdf(dist, params['algorithm']['x'], params['path']['visua'])
#     eps = g.eps
#     params['algorithm']['N_each'] = N_each
#     postproc = PostprocessABC(C_limits, eps, num_bin_joint, params)
#     postproc.calc_marginal_pdf('_calibration_')


g.accepted = np.load(filename_accepted)['C']
g.dist = np.load(filename_accepted)['dist']
num_bin_joint = 50
N_each = 100


if algorithm == 'acc-rej':
    postproc.plot_eps()
    new_eps = 3500
    g.accepted = g.accepted[g.dist < new_eps]
    g.dist = g.dist[g.dist < new_eps]
    logging.info('accepted {} values ({}%)'.format(len(g.accepted),
                                                   round(len(g.accepted) / params['algorithm']['N_total'] * 100, 2)))

accepted = g.accepted[g.accepted[:, 0] < 0.0]
g.dist = g.dist[g.accepted[:, 0] < 0.0]
g.accepted = accepted
logging.info('accepted {} values ({}%)'.format(len(g.accepted),
                                               round(len(g.accepted) / params['algorithm']['N_total'] * 100, 2)))

# C_limits = params['C_limits']
C_limits = np.zeros((10, 2))
C_limits[0] = [np.min(g.accepted[:, 0]), np.max(g.accepted[:, 0])]
C_limits[1] = [np.min(g.accepted[:, 1]), np.max(g.accepted[:, 1])]
C_limits[2] = [np.min(g.accepted[:, 2]), np.max(g.accepted[:, 2])]
if params['model']['N_params'] == 4:
    C_limits[3] = [np.min(g.accepted[:, 3]), np.max(g.accepted[:, 3])]
# C_limits[4] = [np.min(g.accepted[:, 4]), np.max(g.accepted[:, 4])]
# C_limits[5] = [np.min(g.accepted[:, 5]), np.max(g.accepted[:, 5])]
print(C_limits)
eps = g.eps
params['algorithm']['N_each'] = N_each


# N_params = 3
# fig1, axarr1 = plt.subplots(nrows=1, ncols=N_params, figsize=(2*N_params, 2.5))
# fig2, axarr2 = plt.subplots(nrows=1, ncols=N_params, figsize=(2*N_params, 2.5))
# for num_bin_joint in range(10, 50):
#     postproc = PostprocessABC(C_limits, eps, num_bin_joint, params)
#     postproc.calc_final_C()
#     print(postproc.C_final_joint, len(postproc.C_final_joint))
#     print(postproc.C_final_smooth, len(postproc.C_final_smooth))
#
#     for i in range(N_params):
#         axarr1[i].scatter([num_bin_joint] * len(postproc.C_final_joint), np.array(postproc.C_final_joint)[:, i])
#         axarr2[i].scatter([num_bin_joint] * len(postproc.C_final_smooth), np.array(postproc.C_final_smooth)[:, i])
#     for i in range(N_params):
#         axarr1[i].set_xlabel('Number of bins')
#         axarr2[i].set_xlabel('Number of bins')
# axarr1[0].set_ylabel(r'$C_i$')
# axarr2[0].set_ylabel(r'$C_i$')
# fig1.subplots_adjust(left=0.1, right=0.98, hspace=0.3, bottom=0.21, top=0.97)
# fig2.subplots_adjust(left=0.1, right=0.98, hspace=0.3, bottom=0.21, top=0.97)
#
# fig1.savefig(os.path.join(path['visua'], 'Num_bins_joint'))
# fig2.savefig(os.path.join(path['visua'], 'Num_bins_smooth'))
# plt.close('all')


postproc = PostprocessABC(C_limits, eps, num_bin_joint, params)
postproc.calc_final_C()
# postproc.calc_marginal_pdf()
postproc.calc_conditional_pdf()
# # postproc.plot_eps()
# postproc.calc_compare_sum_stat(path['output'], params['compare_pdf']['summary_statistics'], scale='TEST')
# postproc.calc_compare_sum_stat(params['compare_pdf']['summary_statistics'], scale='TEST_M')
