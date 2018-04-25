import gc
import itertools
import logging
import random as rand
from time import time

import global_var as g
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
from matplotlib import animation
import model
import numpy as np
import utils


########################################################################################################################
## class PostprocessABC
########################################################################################################################
class PostprocessABC(object):

    def __init__(self, C_limits, eps, N, folder):

        logging.info('\nPostprocessing')
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
        self.params_names = [r'$C_S$', r'$C_2$', r'$C_3$', r'$C_4$', r'$C_5$', r'$C_6$', r'$C_7$', r'$C_8$', r'$C_9$']

        self.C_final_dist = []
        self.C_final_joint = []
        self.C_final_marginal = np.empty(self.N.params)

    def calc_final_C(self):
        """ Estimate the best fit of parameters.
        For 1 parameter: based on minimum distance;
        For more then 1 parameter: based on joint pdf.
        """

        if self.N.params == 1:
            # C_final_dist only
            self.C_final_dist = [[g.accepted[:, 0][np.argmin(g.dist)]]]
            logging.info('Estimated parameter: {}'.format(self.C_final_dist[0][0]))
            logging.info('Min distance: {}'.format(g.dist[np.argmin(g.dist)]))
            # logging.info('Estimated parameter:{}'.format(np.sqrt(-self.C_final_dist[0][0]/2)))
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
        # g.accepted[:, 0] = np.sqrt(-g.accepted[:, 0]/2)

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
                        # print('{} marginal mean is {} and max is {}'. format(self.params_names[i], mean, max))
                        self.C_final_marginal[i] = max
                        ax = plt.subplot2grid((self.N.params, self.N.params), (i, i))
                        ax.hist(g.accepted[:, i], bins=self.num_bin_joint, normed=1, alpha=0.5, color='grey',
                                range=self.C_limits[i])
                        ax.plot(x, y)
                        # ax.axvline(mean, linestyle='--', color='g', label='mean')
                        # ax.axvline(max, linestyle='--', color='r', label='max')
                        ax.axvline(self.C_final_dist[i], linestyle='--', color='g', label='min dist')
                        if self.C_final_joint:
                            for C in self.C_final_joint:
                                ax.axvline(C[i], linestyle='--', color='b', label='joint max')
                        ax.axis(xmin=self.C_limits[i, 0], xmax=self.C_limits[i, 1])
                        ax.set_xlabel(self.params_names[i])
                    elif i < j:
                        ax = plt.subplot2grid((self.N.params, self.N.params), (i, j))
                        ax.axis(xmin=self.C_limits[j, 0], xmax=self.C_limits[j, 1],
                                ymin=self.C_limits[i, 0], ymax=self.C_limits[i, 1])
                        plt.hist2d(g.accepted[:, j], g.accepted[:, i], bins=self.num_bin_joint, cmap=cmap,
                                   range=[self.C_limits[j], self.C_limits[i]])
            plt.legend(loc='lower left', bbox_to_anchor=(-6.5, 3.5), fancybox=True, shadow=True)
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
            # ax.set_title('KL distance')
            fig.subplots_adjust(left=0.19, right=0.95, bottom=0.15, top=0.9)
            fig.savefig(self.folder + self.params_names[i][1:-1])
        gc.collect()

    def plot_compare_tau(self, MCMC, scale='LES'):

        if self.N.params == 1:
            C_final_dist_new = self.C_final_dist[0].copy()
        else:
            C_final_dist_new = self.C_final_dist.copy()
        C_final_dist_new[0] = -2 * C_final_dist_new[0] ** 2
        C_final_joint = 0
        if len(self.C_final_joint) < 5 and self.N.params != 1:
            # if len(self.C_final_joint) == 1:
            #     C_final_joint = self.C_final_joint[0].copy()
            #     C_final_joint[0] = -2 * C_final_joint[0] ** 2
            # else:
            C_final_joint = self.C_final_joint.copy()
            for i in range(len(C_final_joint)):
                C_final_joint[i][0] = -2 * C_final_joint[i][0] ** 2
        C_final_marginal = self.C_final_marginal

        fig, axarr = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(6.5, 2.5))

        # create pdfs
        if scale == 'LES':
            titles = [r'$\widetilde{\sigma}_{11}$', r'$\widetilde{\sigma}_{12}$', r'$\widetilde{\sigma}_{13}$']
            current_model = model.NonlinearModel(g.LES, 1, self.N, self.C_limits, MCMC)
        if scale == 'TEST_M':
            titles = [r'$\widehat{\sigma}_{11}$', r'$\widehat{\sigma}_{12}$', r'$\widehat{\sigma}_{13}$']
            current_model = model.NonlinearModel(g.TEST_sp, 1, self.N, self.C_limits, MCMC)
        if scale == 'TEST':
            titles = [r'$\widehat{\sigma}_{11}$', r'$\widehat{\sigma}_{12}$', r'$\widehat{\sigma}_{13}$']
            current_model = model.NonlinearModel(g.TEST, 1, self.N, self.C_limits, MCMC)

        tau_modeled_dist = current_model.Reynolds_stresses_from_C_tau(C_final_dist_new)
        tau_modeled_marginal = current_model.Reynolds_stresses_from_C_tau(C_final_marginal)

        for ind, key in enumerate(['uu', 'uv', 'uw']):
            # Plot true pdf
            if scale == 'LES':
                x, y = utils.pdf_from_array_with_x(g.LES.tau_true[key].flatten(), g.bins, g.domain)
                y = utils.take_safe_log(y)
            if scale == 'TEST':
                x, y = utils.pdf_from_array_with_x(g.TEST.tau_true[key].flatten(), g.bins, g.domain)
                y = utils.take_safe_log(y)
            if scale == 'TEST_M':
                x, y = utils.pdf_from_array_with_x(g.TEST_sp.tau_true[key].flatten(), g.bins, g.domain)
                y = utils.take_safe_log(y)
            axarr[ind].plot(x, y, 'r', linewidth=2, label='true')
            axarr[ind].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
            # plot min dist pdf
            x, y = utils.pdf_from_array_with_x(tau_modeled_dist[key].flatten(), g.bins, g.domain)
            y = utils.take_safe_log(y)
            axarr[ind].plot(x, y, 'g', linewidth=2, label='modeled dist')
            # # plot max marginal
            # x, y = utils.pdf_from_array_with_x(tau_modeled_marginal[key].flatten(), g.bins, g.domain)
            # axarr[ind].plot(x, y, 'm', linewidth=2, label='modeled marginal max')
            axarr[ind].set_xlabel(titles[ind])

        # Plot max joint pdf
        if C_final_joint:
            for i in range(len(C_final_joint)):
                tau_modeled_joint = current_model.Reynolds_stresses_from_C_tau(C_final_joint[i])
                y_dict = dict()
                for ind, key in enumerate(['uu', 'uv', 'uw']):
                    x, y_dict[key] = utils.pdf_from_array_with_x(tau_modeled_joint[key].flatten(), g.bins, g.domain)
                    y = utils.take_safe_log(y_dict[key])
                    axarr[ind].plot(x, y, 'b', linewidth=2, label='modeled joint')

                np.savez('./plots/pdf.npz', x=x, uu=y_dict['uu'], uv=y_dict['uv'], uw=y_dict['uw'])

        axarr[0].axis(xmin=g.domain[0], xmax=g.domain[1], ymin=-7)      #ymin=g.TINY_log-0.5)
        axarr[0].set_ylabel('ln(pdf)')
        plt.legend(loc=0)
        fig.subplots_adjust(left=0.1, right=0.95, wspace=0.1, bottom=0.18, top=0.9)
        # axarr[0].set_yscale('log', basey=np.e)
        fig.savefig(self.folder + scale)
        del fig, axarr
        gc.collect()

    def plot_eps(self):
        num_eps = 6
        eps = np.linspace(7.2, 15, num_eps)

        # eps = np.append(8.877, eps)

        C_mean = np.empty((self.N.params, num_eps))
        C_max = np.empty_like(C_mean)
        C_std = np.empty((self.N.params, num_eps))
        C_h = np.empty((self.N.params, num_eps))

        fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(6.5, 2.5))
        for ind, new_eps in enumerate(eps):
            g.accepted = np.load('./plots/accepted.npz')['C']
            g.dist = np.load('./plots/accepted.npz')['dist']
            # g.accepted[:, 0] = np.sqrt(-g.accepted[:, 0] / 2)

            g.accepted = g.accepted[g.dist < new_eps]
            g.dist = g.dist[g.dist < new_eps]
            logging.info('eps = {}: accepted {} values ({}%)'.format(new_eps,
                len(g.accepted), round(len(g.accepted) / (g.N.total) * 100, 2)))
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

    def scatter_animation(self):

        for i in range(self.N.params):
            x = g.accepted[:, i]
            # if i == 0:
            #     print(x)
            #     x = np.sqrt(-x/2)
            fig = plt.figure(figsize=(6.5, 6.5))
            ax = plt.axes()
            # ax.axis(xmin=self.C_limits[i, 0], xmax=self.C_limits[i, 1], ymax=self.eps + 1)
            ax.axis(xmin=0.2, xmax=0.25, ymin=2, ymax=13)
            point, = ax.plot([], [], 'bo', lw=1)

            # for animation
            def init():
                point.set_data([], [])
                return point,

            def animate(i):
                point.set_data(x[:i], g.dist[:i])
                return point,

            # call the animator.  blit=True means only re-draw the parts that have changed.
            anim = animation.FuncAnimation(fig, animate, init_func=init,
                                           frames=len(g.dist), interval=60, blit=True)
            anim.save(self.folder+'basic_animation.mp4', fps=1, extra_args=['-vcodec', 'libx264'])


