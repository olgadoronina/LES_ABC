import gc
import itertools
import logging
import random as rand
from time import time

import global_var as g
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import model
import numpy as np
import utils


class ABC(object):

    def __init__(self, N, C_limits, eps):

        logging.info('ABC algorithm')
        self.N = N
        self.N_total = N.each ** N.params
        self.M = N.training
        self.C_limits = C_limits
        self.eps = eps
        self.C_array = self.form_C_array_manual()

        if N.params == 1 or N.params_in_task == 0:
            self.work_func = work_function_single_value
        else:
            self.work_func = work_function_multiple_values


    def form_C_array_random(self):
        """Create list of lists of N parameters uniformly distributed on given interval
        :return: list of lists of sampled parameters
        """
        C_array = []
        for i in range(self.N.each ** self.N.params):
            C = []
            for j in range(self.N.params):
                c = rand.uniform(self.C_limits[j][0], self.C_limits[j][1])
                C.append(-2*c**2)
            C_array.append(C)

        return C_array

    def form_C_array_manual(self):
        """ Create list of lists of N parameters manually(make grid) uniformly distributed on given interval
        :return: list of lists of sampled parameters
        """
        if self.N.params == 1:
            C_array = []
            C1 = np.linspace(self.C_limits[0][0], self.C_limits[0][1], self.N.each + 1)
            C1 = C1[:-1] + (C1[1] - C1[0]) / 2
            C1 = -2 * C1 ** 2
            for i in C1:
                C_array.append([i])
        else:
            C = np.ndarray((self.N.params, self.N.each))
            for i in range(self.N.params):
                C[i, :] = utils.uniform_grid(self.C_limits[i], self.N.each)
            C[0] = -2 * C[0] ** 2
            permutation = itertools.product(*C)
            C_array = list(map(list, permutation))
        return C_array

    def main_loop(self):
        """ Main loop of ABC algorithm, fill list of accepted parameters
        and list of distances (of accepted parameters)"""
        start = time()
        result = []
        if g.par_process:
            g.par_process.run(func=self.work_func, tasks=self.C_array)
            result = g.par_process.get_results()
        else:
            from tqdm import tqdm
            with tqdm(total=self.N.each ** (self.N.params - self.N.params_in_task)) as pbar:
                for C in self.C_array:
                    result.append(self.work_func(C))
                    pbar.update()
            pbar.close()
        end = time()
        utils.timer(start, end, 'Time ')
        if self.N.params_in_task == 0:
            g.accepted = np.array([C[:self.N.params] for C in result if C])
            g.dist = np.array([C[-1] for C in result if C])
        else:
            g.accepted = np.array([chunk[:self.N.params] for item in result for chunk in item])
            g.dist = np.array([chunk[-1] for item in result for chunk in item])
        g.accepted[:, 0] = np.sqrt(-g.accepted[:, 0] / 2)  # return back to standard Cs (-2*Cs^2)
        # np.savetxt('accepted_'+str(N_params_in_task)+'.out', g.accepted)
        # np.savetxt('dist_'+str(N_params_in_task)+'.out', g.dist)
        logging.info('Number of accepted values: {} {}%'.format(len(g.accepted),
                                                                round(len(g.accepted) / self.N_total * 100, 2)))


########################################################################################################################
## Distance functions
########################################################################################################################
def distance_between_pdf_KL(pdf_modeled, key, axis=1):
    """Calculate statistical distance between two pdf as
    the Kullback-Leibler (KL) divergence (no symmetry).
    Function for N_params_in_task > 0
    :param pdf_modeled: array of modeled pdf
    :param key: tensor component(key of dict)
    :return: 1D array of calculated distance
    """
    log_fill = np.empty_like(pdf_modeled)
    log_fill.fill(g.TINY_log)
    log_modeled = np.log(pdf_modeled, out=log_fill, where=pdf_modeled > g.TINY)
    dist= np.sum(np.multiply(g.TEST_sp.tau_pdf_true[key], (g.TEST_sp.log_tau_pdf_true[key] - log_modeled)), axis=axis)
    # dist = np.sum(np.multiply(pdf_modeled, (log_modeled - g.TEST_sp.log_tau_pdf_true[key])), axis=axis)

    return dist


def distance_between_pdf_L1log(pdf_modeled, key, axis=1):
    """Calculate statistical distance between two pdf as
    :param pdf_modeled: array of modeled pdf
    :return:            scalar of calculated distance
    """
    log_fill = np.empty_like(pdf_modeled)
    log_fill.fill(g.TINY_log)
    log_modeled = np.log(pdf_modeled, out=log_fill, where=pdf_modeled > g.TINY)
    dist = 0.5 * np.sum(np.abs(log_modeled - g.TEST_sp.log_tau_pdf_true[key]), axis=axis)
    return dist


def distance_between_pdf_LSE(pdf_modeled, key, axis=1):
    """ Calculate statistical distance between two pdf as mean((P1-P2)^2).
    :param pdf_modeled: array of modeled pdf
    :param key: tensor component(key of dict)
    :param axis: equal 1 when pdf_modeled is 2D array
    :return: scalar or 1D array of calculated distance
    """
    dist = np.mean((pdf_modeled - g.TEST_sp.tau_pdf_true[key]) ** 2, axis=axis)
    return dist


def distance_between_pdf_L2(pdf_modeled, key, axis=1):
    """ Calculate statistical distance between two pdf as sqrt(sum((P1-P2)^2)).
    :param pdf_modeled: array of modeled pdf
    :param key: tensor component(key of dict)
    :param axis: equal 1 when pdf_modeled is 2D array
    :return: scalar or 1D array of calculated distance
    """
    dist = np.sqrt(np.sum((pdf_modeled - g.TEST_sp.tau_pdf_true[key]) ** 2, axis=axis))
    return dist


def distance_between_pdf_LSElog(pdf_modeled, key, axis=1):
    """ Calculate statistical distance between two pdf as mean((ln(P1)-ln(P2))^2).
    :param pdf_modeled: array of modeled pdf
    :param key: tensor component(key of dict)
    :return: 1D array of calculated distance
    """
    log_fill = np.empty_like(pdf_modeled)
    log_fill.fill(g.TINY_log)
    log_modeled = np.log(pdf_modeled, out=log_fill, where=pdf_modeled > g.TINY)
    dist = np.mean((log_modeled - g.TEST_sp.log_tau_pdf_true[key]) ** 2, axis=axis)
    return dist


def distance_between_pdf_L2log(pdf_modeled, key, axis=1):
    """ Calculate statistical distance between two pdf.
    :param pdf_modeled:
    :param key:
    :param axis:
    :return:
    """
    log_fill = np.empty_like(pdf_modeled)
    log_fill.fill(g.TINY_log)
    log_modeled = np.log(pdf_modeled, out=log_fill, where=pdf_modeled > g.TINY)
    dist = np.sqrt(np.sum((log_modeled - g.TEST_sp.log_tau_pdf_true[key]) ** 2, axis=axis))
    return dist


########################################################################################################################
# Work_functions
########################################################################################################################
def work_function_single_value(C):
    """ Worker function for parallel regime (for pool.map from multiprocessing module)
    :param C: list of sampled parameters
    :return:  list[bool, Cs, dist], where bool=True, if values are accepted
    """
    tau = g.TEST_Model.Reynolds_stresses_from_C(C)

    dist = 0
    for key in g.TEST_Model.elements_in_tensor:
        pdf = np.histogram(tau[key].flatten(), bins=g.bins, range=g.domain, normed=1)[0]
        d = distance_between_pdf_KL(pdf_modeled=pdf, key=key, axis=0)
        dist += d
    if dist <= g.eps:
        result = C[:]
        result.append(dist)
        return result


def work_function_multiple_values(C):
    """ Worker function for parallel regime (for pool.map from multiprocessing module)
    :param C: list of sampled parameters
    :return:  list[bool, Cs, dist], where bool=True, if values are accepted
    """
    return g.TEST_Model.Reynolds_stresses_from_C(C, distance_between_pdf_KL)


########################################################################################################################
## class PostprocessABC
########################################################################################################################
class PostprocessABC(object):

    def __init__(self, C_limits, eps, N, folder):

        logging.info('Postprocessing')
        if len(g.accepted) == 0:
            logging.error('Oops! No accepted values')
            exit()
        self.N = N
        self.num_bin_joint = N.bin_joint
        self.folder = folder
        if self.N.params != len(g.accepted[0]):
            self.N.params = len(g.accepted[0])
            logging.warning('Wrong number of params in params.py. Use {} params'.format(self.N.params))
        self.C_limits = C_limits
        self.eps = eps
        self.params_names = [r'$C_s$', r'$C_2$', r'$C_3$', r'$C_4$', r'$C_5$', r'$C_6$', r'$C_7$', r'$C_8$', r'$C_9$']

        self.C_final_dist = []
        self.C_final_joint = []

    def calc_final_C(self):
        """ Estimate the best fit of parameters.
        For 1 parameter: based on minimum distance;
        For more then 1 parameter: based on joint pdf.
        """

        if self.N.params == 1:
            # C_final_dist only
            self.C_final_dist = [[g.accepted[:, 0][np.argmin(g.dist)]]]
            logging.info('Estimated parameter:{}'.format(np.sqrt(-self.C_final_dist[0][0]/2)))
        else:
            # C_final_dist
            minim = np.argmin(g.dist)
            self.C_final_dist = g.accepted[minim, :]
            C_final_dist = self.C_final_dist
            # C_final_dist[0] = np.sqrt(-C_final_dist[0] / 2)
            logging.info('Minimum distance is {} in: {}'.format(g.dist[minim], C_final_dist))
            # C_final_joint
            H, edges = np.histogramdd(g.accepted, bins=self.num_bin_joint)
            logging.debug('Max number in bin: ' + str(np.max(H)))
            logging.debug('Mean number in bin: ' + str(np.mean(H)))
            edges = np.array(edges)
            C_bin = (edges[:, :-1] + edges[:, 1:]) / 2  # shift value in the center of the bin
            ind = np.argwhere(H == np.max(H))
            for i in ind:
                point = []
                for j in range(self.N.params):
                    point.append(C_bin[j, i[j]])
                self.C_final_joint.append(point)
            if len(ind) > 10:
                logging.warning('Can not estimate parameters from joint pdf!'
                                'Too many bins ({} bins, max value {}) '
                                'with the same max value in joint pdf'.format(len(ind), np.max(H)))
            else:
                logging.info('Estimated parameters from joint pdf: {}'.format(self.C_final_joint))

    def plot_marginal_pdf(self):

        # Uncomment to make figure for each marginal pdf
        # for i in range(self.N.params):
        #     plt.hist(self.accepted[:, i], bins=N_each, normed=1, range=C_limits[i])
        #     plt.xlabel(params_names[i])
        #     plt.show()

        # accepted = g.accepted
        # accepted[:, 0] = np.sqrt(accepted[:, 0]/2)
        if self.N.params > 1:
            cmap = plt.cm.jet  # define the colormap
            cmaplist = [cmap(i) for i in range(cmap.N)]  # extract all colors from the .jet map
            cmaplist[0] = ('white')  # force the first color entry to be grey
            cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

            fig = plt.figure(figsize=(6.5, 6.5))
            for i in range(self.N.params):
                for j in range(self.N.params):
                    if i == j:
                        mean = np.mean(g.accepted[:, i])
                        x, y = utils.pdf_from_array_with_x(g.accepted[:, i], bins=self.N.each, range=self.C_limits[i])
                        max = x[np.argmax(y)]
                        ax = plt.subplot2grid((self.N.params, self.N.params), (i, i))
                        ax.hist(g.accepted[:, i], bins=self.num_bin_joint, normed=1, alpha=0.5, color='grey',
                                range=self.C_limits[i])
                        ax.plot(x, y)
                        ax.axvline(mean, linestyle='--', color='g', label='mean')
                        ax.axvline(max, linestyle='--', color='r', label='max')
                        ax.axis(xmin=self.C_limits[i, 0], xmax=self.C_limits[i, 1])
                        ax.set_xlabel(self.params_names[i])
                    elif i < j:
                        ax = plt.subplot2grid((self.N.params, self.N.params), (i, j))
                        ax.axis(xmin=self.C_limits[j, 0], xmax=self.C_limits[j, 1], ymin=self.C_limits[i, 0], ymax=self.C_limits[i, 1])
                        plt.hist2d(g.accepted[:, j], g.accepted[:, i], bins=self.num_bin_joint, cmap=cmap,
                                   range=[self.C_limits[j], self.C_limits[i]])
            plt.legend(loc='lower left', bbox_to_anchor=(-2.5, 0.35), fancybox=True, shadow=True)
            fig.subplots_adjust(left=0.05, right=0.98, wspace=0.25, bottom=0.08, top=0.95)
            # fig.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.5)
            fig.savefig(self.folder+'marginal')
            del fig

    def plot_scatter(self):

        for i in range(self.N.params):
            x = g.accepted[:, i]
            # if i == 0:
            #     print(x)
            #     x = np.sqrt(-x/2)
            fig = plt.figure(figsize=(3.2, 2.8))
            ax = plt.axes()
            ax.axis(xmin=self.C_limits[i, 0], xmax=self.C_limits[i, 1], ymax=self.eps + 1)
            ax.scatter(x, g.dist, color='blue')
            ax.set_xlabel(self.params_names[i])
            ax.set_ylabel(r'$\sum_{i,j}\rho(\mathcal{S}_{ij}^{\mathcal{F}},\mathcal{S}_{ij})$')
            ax.set_title('KL distance')
            fig.subplots_adjust(left=0.19, right=0.95, bottom=0.15, top=0.9)
            fig.savefig(self.folder + self.params_names[i][1:-1])
        gc.collect()

    def plot_compare_tau(self, scale='LES'):

        C_final_dist_new = self.C_final_dist
        C_final_dist_new[0] = -2 * C_final_dist_new[0] ** 2
        fig, axarr = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(6.5, 2.5))
        if scale == 'LES':
            titles = [r'$\widetilde{\sigma}_{11}$', r'$\widetilde{\sigma}_{12}$', r'$\widetilde{\sigma}_{13}$']
            if len(self.C_final_joint) == 1 and self.N.params != 1:
                tau_modeled_joint = model.NonlinearModel(g.LES, self.N).Reynolds_stresses_from_C(self.C_final_joint[0])
            tau_modeled_dist = model.NonlinearModel(g.LES, self.N, self.C_limits).Reynolds_stresses_from_C(
                C_final_dist_new)
        if scale == 'TEST_M':
            titles = [r'$\widehat{\sigma}_{11}$', r'$\widehat{\sigma}_{12}$', r'$\widehat{\sigma}_{13}$']
            if len(self.C_final_joint) == 1 and self.N.params != 1:
                tau_modeled_joint = model.NonlinearModel(g.TEST_sp, self.N).Reynolds_stresses_from_C(
                    self.C_final_joint[0])
            tau_modeled_dist = model.NonlinearModel(g.TEST_sp, self.N, self.C_limits).Reynolds_stresses_from_C(
                C_final_dist_new)
        if scale == 'TEST':
            titles = [r'$\widehat{\sigma}_{11}$', r'$\widehat{\sigma}_{12}$', r'$\widehat{\sigma}_{13}$']
            if len(self.C_final_joint) == 1 and self.N.params != 1:
                tau_modeled_joint = model.NonlinearModel(g.TEST, self.N).Reynolds_stresses_from_C(self.C_final_joint[0])
            tau_modeled_dist = model.NonlinearModel(g.TEST, self.N, self.C_limits).Reynolds_stresses_from_C(
                C_final_dist_new)
        for ind, key in enumerate(['uu', 'uv', 'uw']):
            if scale == 'LES':
                x, y = utils.pdf_from_array_with_x(g.LES.tau_true[key].flatten(), g.bins, g.domain)
            if scale == 'TEST':
                x, y = utils.pdf_from_array_with_x(g.TEST.tau_true[key].flatten(), g.bins, g.domain)
            if scale == 'TEST_M':
                x, y = utils.pdf_from_array_with_x(g.TEST_sp.tau_true[key].flatten(), g.bins, g.domain)
            axarr[ind].plot(x, y, 'r', linewidth=2, label='true')
            axarr[ind].xaxis.set_major_locator(ticker.MultipleLocator(0.5))

            if len(self.C_final_joint) == 1 and self.N.params != 1:
                x, y = utils.pdf_from_array_with_x(tau_modeled_joint[key].flatten(), g.bins, g.domain)
                axarr[ind].plot(x, y, 'b', linewidth=2, label='modeled joint')
            x, y = utils.pdf_from_array_with_x(tau_modeled_dist[key].flatten(), g.bins, g.domain)
            axarr[ind].plot(x, y, 'g', linewidth=2, label='modeled dist')
            axarr[ind].set_xlabel(titles[ind])
        axarr[0].axis(xmin=g.domain[0], xmax=g.domain[1], ymin=1e-5)
        axarr[0].set_ylabel('pdf')
        axarr[0].set_yscale('log', nonposy='clip')
        plt.legend(loc=0)
        fig.subplots_adjust(left=0.1, right=0.95, wspace=0.1, bottom=0.18, top=0.9)
        fig.savefig(self.folder + scale)
        del fig, axarr
        gc.collect()

    def plot_eps(self):
        num_eps = 6
        eps = np.linspace(15, 40, num_eps)

        eps = np.append(8.877, eps)

        C_mean = np.empty((self.N.params, num_eps))
        C_max = np.empty_like(C_mean)
        C_std = np.empty((self.N.params, num_eps))
        C_h = np.empty((self.N.params, num_eps))

        fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(6.5, 2.5))
        for ind, new_eps in enumerate(eps):
            g.accepted = np.load('./plots/accepted.npz')['C']
            g.dist = np.load('./plots/accepted.npz')['dist']
            g.accepted[:, 0] = np.sqrt(-g.accepted[:, 0] / 2)

            g.accepted = g.accepted[g.dist < new_eps]
            g.dist = g.dist[g.dist < new_eps]
            logging.info('accepted {} values ({}%)'.format(len(g.accepted), round(len(g.accepted) /
                                                                                  (self.N.each**self.N.params) * 100, 2)))
            for i in range(self.N.params):
                data = g.accepted[:, i]
                C_std[i, ind] = np.std(data)
                C_mean[i, ind], C_h[i, ind] = utils.mean_confidence_interval(data, confidence=0.95)

                x, y = utils.pdf_from_array_with_x(data, bins=self.N.each, range=self.C_limits[i])
                C_max[i, ind] = x[np.argmax(y)]
                axarr[i].plot(x, y, label=r'$\epsilon = {}$'.format(new_eps))
                axarr[i].set_xlabel(self.params_names[i])

        axarr[0].set_ylabel('marginal pdf')
        # Put a legend below current axis
        legend = plt.legend(loc='upper center', bbox_to_anchor=(1., 1.1))
        frame = legend.get_frame()
        frame.set_alpha(1)
        fig.subplots_adjust(left=0.08, right=0.9, wspace=0.17, bottom=0.17, top=0.9)
        fig.savefig(self.folder + 'eps_marginal')

        fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(6.5, 2.5))
        for i in range(self.N.params):
            axarr[i].plot(eps, C_mean[i], 'b.-', label='mean')
            axarr[i].plot(eps, C_max[i], 'g.-', label='max')
            axarr[i].set_title(self.params_names[i])
            axarr[i].set_xlabel('epsilon')
            axarr[i].xaxis.set_major_locator(ticker.MultipleLocator(5))
        axarr[0].set_ylabel(r'$C_i$')
        plt.legend(loc=0)
        fig.subplots_adjust(left=0.1, right=0.97, wspace=0.4, bottom=0.2, top=0.85)
        fig.savefig(self.folder + 'eps_plot')



        fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(6.5, 2.5))
        for i in range(self.N.params):
            axarr[i].plot(eps, C_std[i], 'b.-')
            axarr[i].set_title(self.params_names[i])
            axarr[i].set_xlabel('epsilon')
            axarr[i].xaxis.set_major_locator(ticker.MultipleLocator(5))
            # axarr[i].axis(ymin=np.min(C_mean[i])-0.01*, ymax=np.max(C_mean[i])+0.1)
        axarr[0].set_ylabel(r'std($C_i$)')
        fig.subplots_adjust(left=0.1, right=0.97, wspace=0.4, bottom=0.2, top=0.85)
        fig.savefig(self.folder + 'eps_std')


        fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(6.5, 2.5))
        for i in range(self.N.params):
            axarr[i].plot(eps, C_h[i], 'b.-')
            axarr[i].set_title(self.params_names[i])
            axarr[i].set_xlabel('epsilon')
            axarr[i].xaxis.set_major_locator(ticker.MultipleLocator(5))
            # axarr[i].axis(ymin=np.min(C_mean[i])-0.01*, ymax=np.max(C_mean[i])+0.1)
        axarr[0].set_ylabel(r'$95\%$ confident interval')
        fig.subplots_adjust(left=0.12, right=0.97, wspace=0.4, bottom=0.2, top=0.85)
        fig.savefig(self.folder + 'eps_h')

