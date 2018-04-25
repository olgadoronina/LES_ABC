import gc
import glob
import logging
from math import pi

import global_var as g
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
import numpy as np
import utils


class Plot(object):
    def __init__(self, folder, plot):
        self.folder = folder
        self.plot_info = plot
        self.map_bounds = None

    def imagesc(self, Arrays, map_bounds, titles, name=None):

        cmap = plt.cm.jet  # define the colormap
        cmaplist = [cmap(i) for i in range(cmap.N)]  # extract all colors from the .jet map
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)  # create the new map
        norm = mpl.colors.BoundaryNorm(map_bounds, cmap.N)

        axis = [0, 2 * pi, 0, 2 * pi]
        if len(Arrays) > 1:
            fig, axes = plt.subplots(nrows=1, ncols=len(Arrays), sharey=True, figsize=(6.5, 3))
            k = 0
            for ax in axes.flat:
                im = ax.imshow(Arrays[k].T, origin='lower', cmap=cmap, norm=norm, interpolation="nearest", extent=axis)
                ax.set_title(titles[k])
                ax.set_adjustable('box-forced')
                ax.set_xlabel(r'$x$')
                ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
                k += 1
            axes[0].set_ylabel(r'$y$')
            cbar_ax = fig.add_axes([0.89, 0.18, 0.017, 0.68])  # ([0.85, 0.15, 0.05, 0.68])
            fig.subplots_adjust(left=0.07, right=0.87, wspace=0.1, bottom=0.2, top=0.9)
            fig.colorbar(im, cax=cbar_ax, ax=axes.ravel().tolist())
        else:
            fig = plt.figure(figsize=(6.5, 5))
            ax = plt.gca()
            im = ax.imshow(Arrays[0].T, origin='lower', cmap=cmap, norm=norm, interpolation="nearest")
            plt.colorbar(im, fraction=0.05, pad=0.04)
        if name:
            # pickle.dump(ax, open(self.folder + name, 'wb'))
            fig.savefig(self.folder + name)
        del ax, im, fig, cmap
        gc.collect()

    # def histogram(field, bins, pdf=None, label=None, log=False):
    #     plt.figure(figsize=(6, 4))
    #     plt.hist(field, bins=bins, alpha=0.4, normed=1)
    #
    #     # h, edges = np.histogram(field, bins=bins, range=[-2,2], normed=1)
    #     if pdf:
    #         x = np.linspace(min(field), max(field), 100)
    #         mu = np.mean(field)
    #         sigma = np.std(field)
    #         plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r--', linewidth=3, label='Gaussian')
    #         plt.legend(loc=0)
    #     if log:
    #         plt.yscale('log', nonposy='clip')
    #     if label:
    #         plt.xlabel(label)
    #         plt.ylabel('pdf(' + label + ')')
    #     plt.axis(xmin=np.min(field), xmax=np.max(field))  # xmax=4xmax = np.max(field)
    #     plt.show()
    #     gc.collect()

    ########################################################################################################################
    # Plot initial data info
    ########################################################################################################################

    def compare_filter_fields(self, hit_data, les_data, test_data):

        # if not g.HIT or not g.LES or not g.TEST:
        #     logging.warning('Can not plot fields: some of them is None')
        if test_data:
            self.imagesc([hit_data['v'][:, :, 127], les_data['v'][:, :, 127], test_data['v'][:, :, 127]],
                         self.map_bounds, name='compare_velocity',
                         titles=[r'$v$', r'$\widetilde{v}$', r'$\widehat{\widetilde{v}}$'])
        else:
            self.imagesc([hit_data['v'][:, :, 127], les_data['v'][:, :, 127]],
                         self.map_bounds, name='compare_velocity',
                         titles=[r'$v$', r'$\widetilde{v}$'])

    def vel_fields(self, scale='LES', dns=None):

        if scale == 'LES':
            if dns:
                titles = [r'$u$', r'$v$', r'$w$']
            else:
                titles = [r'$\widetilde{u}$', r'$\widetilde{v}$', r'$\widetilde{w}$']
            self.imagesc([g.LES.field['u'][:, :, 127], g.LES.field['v'][:, :, 127], g.LES.field['w'][:, :, 127]],
                         self.map_bounds, name='LES_velocities', titles=titles)
        elif scale == 'TEST':
            if dns:
                titles = [r'$\widetilde{u}$', r'$\widetilde{v}$', r'$\widetilde{w}$']
            else:
                titles = [r'$\widehat{\widetilde{u}}$', r'$\widehat{\widetilde{v}}$', r'$\widehat{\widetilde{w}}$']
            self.imagesc([g.TEST.field['u'][:, :, 127], g.TEST.field['v'][:, :, 127], g.TEST.field['w'][:, :, 127]],
                         self.map_bounds, name='TEST_velocities', titles=titles)

    def sigma_field(self, scale='LES', dns=None):

        map_bounds = np.linspace(-0.2, 0.2, 9)
        if scale == 'LES':
            if not dns:
                tau = g.LES.tau_true
                name = 'sigma_LES'
                titles = [r'$\sigma_{11}$', r'$\sigma_{12}$', r'$\sigma_{13}$']
                self.imagesc([tau['uu'][:, :, 127], tau['uv'][:, :, 127], tau['uw'][:, :, 127]],
                            map_bounds, name=name, titles=titles)
        elif scale == 'TEST':
            tau = g.TEST.tau_true
            name = 'sigma_TEST'
            if dns:
                titles = [r'$\sigma_{11}$', r'$\sigma_{12}$', r'$\sigma_{13}$']
            else:
                titles = [r'$\widehat{\sigma}_{11}$', r'$\widehat{\sigma}_{12}$', r'$\widehat{\sigma}_{13}$']

            self.imagesc([tau['uu'][:, :, 127], tau['uv'][:, :, 127], tau['uw'][:, :, 127]],
                        map_bounds, name=name, titles=titles)

    def sigma_pdf(self, dns=None):

        name = 'sigma_pdf'
        fig, axarr = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(6.5, 2.4))
        if dns:
            titles = [r'$\sigma_{11}$', r'$\sigma_{12}$', r'$\sigma_{13}$']
            labels = ['DNS', 'LES']
        else:
            titles = [r'$\sigma_{11},\ \widehat{\sigma}_{11}$',
                    r'$\sigma_{12},\ \widehat{\sigma}_{12}$',
                    r'$\sigma_{13},\ \widehat{\sigma}_{13}$']
            labels = ['LES', 'test']
        if not dns:
            tau = g.LES.tau_true
            for ind, i in enumerate(['uu', 'uv', 'uw']):
                data = tau[i].flatten()
                x, y = utils.pdf_from_array_with_x(data, g.bins, g.domain)
                axarr[ind].plot(x, y, 'r', linewidth=2, label=labels[0])
                axarr[ind].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
                # axarr[ind].yaxis.set_major_locator(ticker.MultipleLocator())

        tau = g.TEST.tau_true
        for ind, i in enumerate(['uu', 'uv', 'uw']):
            data = tau[i].flatten()
            x, y = utils.pdf_from_array_with_x(data, g.bins, g.domain)
            axarr[ind].plot(x, y, 'g', linewidth=2, label=labels[1])
            axarr[ind].set_xlabel(titles[ind])

        axarr[0].axis(xmin=-1.1, xmax=1.1, ymin=1e-5)
        axarr[0].set_ylabel('pdf')
        axarr[0].set_yscale('log')
        plt.legend(loc=0)
        fig.subplots_adjust(left=0.1, right=0.95, wspace=0.1, bottom=0.2, top=0.9)
        fig.savefig(self.folder+name)
        del fig, axarr
        gc.collect()

    def S_pdf(self):

        name = 'S_pdf_' + str(g.TEST_sp.M)
        fig, axarr = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(6.5, 2.5))
        titles = [r'$\widetilde{S}_{11}/\widehat{\widetilde{S}}_{11}$',
                  r'$\widetilde{S}_{12}/\widehat{\widetilde{S}}_{12}$',
                  r'$\widetilde{S}_{13}/\widehat{\widetilde{S}}_{13}$']
        strain = g.LES.S
        for ind, i in enumerate(['uu', 'uv', 'uw']):
            data = strain[i].flatten()
            x, y = utils.pdf_from_array_with_x(data, g.bins, [-2, 2])
            axarr[ind].plot(x, y, 'r', linewidth=1, label='LES')
            axarr[ind].set_xlabel(titles[ind])
        strain = g.TEST.S
        for ind, i in enumerate(['uu', 'uv', 'uw']):
            data = strain[i].flatten()
            x, y = utils.pdf_from_array_with_x(data, g.bins, [-2, 2])
            axarr[ind].plot(x, y, 'g', linewidth=1, label='test')
        strain = g.TEST_sp.S
        for ind, i in enumerate(['uu', 'uv', 'uw']):
            data = strain[i].flatten()
            x, y = utils.pdf_from_array_with_x(data, g.bins, [-2, 2])
            axarr[ind].plot(x, y, 'b', linewidth=1, label='test sparse M='+str(g.TEST_sp.M))

        # axarr[0].axis(xmin=-1.1, xmax=1.1, ymin=1e-5)
        axarr[0].set_ylabel('pdf')
        plt.legend(loc=0)
        fig.subplots_adjust(left=0.1, right=0.95, wspace=0.1, bottom=0.2, top=0.9)
        fig.savefig(self.folder + name)
        del fig, axarr
        gc.collect()

    def A_compare(self):

        name = 'A_compare_'+str(g.TEST_sp.M)

        fig, axarr = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(6.5, 2.5))
        titles = [r'$\widetilde{A}_{11}/\widehat{\widetilde{A}}_{11}$',
                  r'$\widetilde{A}_{12}/\widehat{\widetilde{A}}_{12}$',
                  r'$\widetilde{A}_{13}/\widehat{\widetilde{A}}_{13}$']
        deriv = g.LES.A
        for ind, i in enumerate(['uu', 'uv', 'uw']):
            y = deriv[i][:, 128, 128]
            x = np.linspace(0, 2 * pi - 2 * pi / 256, 256)
            axarr[ind].plot(x, y, 'r', linewidth=1, label='LES')
            axarr[ind].set_xlabel('$x$')
            axarr[ind].set_title(titles[ind])
        deriv = g.TEST.A
        for ind, i in enumerate(['uu', 'uv', 'uw']):
            y = deriv[i][:, 128, 128]
            x = np.linspace(0, 2 * np.pi, 256)
            axarr[ind].plot(x, y, 'g', linewidth=1, label='test')

        deriv = g.TEST_sp.A
        for ind, i in enumerate(['uu', 'uv', 'uw']):
            x = np.linspace(0, 2 * pi - 2 * pi / g.TEST_sp.M, g.TEST_sp.M)
            y = deriv[i][:, int(g.TEST_sp.M / 2), int(g.TEST_sp.M / 2)]
            axarr[ind].plot(x, y, 'b', linewidth=1, label='test sparse M='+str(g.TEST_sp.M))

        axarr[0].axis(xmin=0, xmax=2 * np.pi, ymin=-7, ymax=7)
        axarr[0].set_ylabel('$A_{ij}$')
        plt.legend(loc=0)
        fig.subplots_adjust(left=0.1, right=0.95, wspace=0.1, bottom=0.16, top=0.83)
        fig.savefig(self.folder + name)
        del fig, axarr
        gc.collect()

    def spectra(self):

        fig = plt.figure(figsize=(4, 3))
        ax = plt.gca()
        files = glob.glob(self.folder + '*.spectra')

        labels = ['DNS', 'LES', 'test']

        for k in range(len(files)):
            f = open(files[k], 'r')
            data = np.array(f.readlines()).astype(np.float)
            x = np.arange(len(data))
            ax.loglog(x, data, '-', linewidth=2, label=labels[k])

        y = 7.2e14 * np.power(x, -5 / 3)
        ax.loglog(x, y, 'r--', label=r'$-5/3$ slope')
        ax.set_title('Spectra')
        ax.set_ylabel(r'$E$')
        ax.set_xlabel(r'k')
        ax.axis(ymin=1e6)
        plt.legend(loc=0)

        fig.subplots_adjust(left=0.16, right=0.95, bottom=0.2, top=0.87)
        fig.savefig(self.folder + 'spectra')


def TS(TS):
    plt.figure(figsize=(5, 5))
    plt.hist(TS.flatten(), bins=500, normed=1, alpha=0.4)
    plt.yscale('log', nonposy='clip')
    plt.xlabel(r'$T_{ij}S^T_{ij}$')
    plt.ylabel(r'pdf($T_{ij}S^T_{ij}$)')
    plt.axis(xmin=-5, xmax=4, ymin=1e-5)
    plt.show()
    gc.collect()


def tau_abc(Cs_abc):
    fig, axarr = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(18, 6))
    titles = [r'$\widehat{T}_{11}$', r'$\widehat{T}_{12}$', r'$\widehat{T}_{13}$']

    x = np.linspace(g.domain[0], g.domain[1], g.bins)
    plots = []
    labels = []
    for ind, i in enumerate(['uu', 'uv', 'uv']):
        line = axarr[ind].plot(x, g.TEST.tau_pdf_true[i], linewidth=3, label='true pdf')
        if ind == 0:
            plots.append(line)
            labels.append('true pdf')
        axarr[ind].set_xlabel(titles[ind])
    for C_s in Cs_abc:
        tau = g.TEST.Reynolds_stresses_from_Cs(C_s)
        for ind, i in enumerate(['uu', 'uv', 'uv']):
            x, y = utils.pdf_from_array(tau[i].flatten(), g.bins, g.domain)
            line = axarr[ind].plot(x, y, linewidth=1, label=r'$C_s \approx\  $' + str(round(C_s, 3)))
            if ind == 0:
                plots.append(line)
                labels.append(r'$C_s \approx $' + str(round(C_s, 3)))
    axarr[0].axis(xmin=-1.1, xmax=1.1, ymin=1e-5)
    axarr[0].set_ylabel('pdf')
    axarr[0].set_yscale('log', nonposy='clip')
    # Shrink current axis's width
    for i in range(3):
        box = axarr[i].get_position()
        axarr[i].set_position([box.x0, box.y0, box.width * 0.9, box.height])
    # Put a legend below current axis
    plt.legend(loc='upper center', bbox_to_anchor=(1.6, 1.1), fancybox=True, shadow=True, ncol=2)
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig('tau_abc.eps')
    del fig, axarr
    gc.collect()


def dist_pdf(dist):
    x = 0.1
    plt.figure(figsize=(3, 3))
    plt.hist(dist, bins=100, normed=1, alpha=0.4)
    # plt.yscale('log', nonposy='clip')
    eps = np.percentile(dist, q=int(x * 100))
    plt.axvline(eps, label='eps')
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'pdf($\rho$)')
    # plt.axis(xmin=-5, xmax=4, ymin=1e-5)
    plt.show()
    gc.collect()




