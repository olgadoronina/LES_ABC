import gc
import logging
from math import pi
import glob, os

import global_var as g
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import utils


class Plot(object):
    def __init__(self, folder):
        self.folder = folder

        self.map_bounds = None

    def imagesc(self, Arrays, map_bounds, titles, name=None):

        cmap = plt.cm.jet  # define the colormap
        cmaplist = [cmap(i) for i in range(cmap.N)]  # extract all colors from the .jet map
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)  # create the new map
        norm = mpl.colors.BoundaryNorm(map_bounds, cmap.N)

        axis = [0, 2 * pi, 0, 2 * pi]
        if len(Arrays) > 1:
            fig, axes = plt.subplots(nrows=1, ncols=len(Arrays), sharey=True, figsize=(6.5, 2.4))
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
            fig.subplots_adjust(left=0.07, right=0.87, wspace=0.1, bottom=0.05, top=0.98)
            fig.colorbar(im, cax=cbar_ax, ax=axes.ravel().tolist())
        else:
            fig = plt.figure(figsize=(6.5, 5))
            ax = plt.gca()
            im = ax.imshow(Arrays[0].T, origin='lower', cmap=cmap, norm=norm, interpolation="nearest")
            plt.colorbar(im, fraction=0.05, pad=0.04)
        if name:
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

        if not g.HIT or not g.LES or not g.TEST:
            logging.warning('Can not plot fields: some of them is None')
        self.imagesc([hit_data['v'][:, :, 127], les_data['v'][:, :, 127], test_data['v'][:, :, 127]],
                     self.map_bounds, name='compare_velocity',
                     titles=[r'$v$', r'$\widetilde{v}$', r'$\widehat{\widetilde{v}}$'])

    def vel_fields(self, scale='LES'):

        if scale == 'LES':
            self.imagesc([g.LES.field['u'][:, :, 127], g.LES.field['v'][:, :, 127], g.LES.field['w'][:, :, 127]],
                         self.map_bounds, name='LES_velocities',
                         titles=[r'$\widetilde{u}$', r'$\widetilde{v}$', r'$\widetilde{w}$'])
        elif scale == 'TEST':
            self.imagesc([g.TEST.field['u'][:, :, 127], g.TEST.field['v'][:, :, 127], g.TEST.field['w'][:, :, 127]],
                         self.map_bounds, name='TEST_velocities',
                         titles=[r'$\widehat{\widetilde{u}}$', r'$\widehat{\widetilde{v}}$',
                                 r'$\widehat{\widetilde{w}}$'])

    def sigma_field(self, scale='LES'):

        map_bounds = np.linspace(-0.2, 0.2, 10)
        if scale == 'LES':
            tau = g.LES.tau_true
            name = 'sigma_LES'
            titles = [r'$\sigma_{11}$', r'$\sigma_{12}$', r'$\sigma_{13}$']

        elif scale == 'TEST':
            tau = g.TEST.tau_true
            name = 'sigma_TEST'
            titles = [r'$\widehat{\sigma}_{11}$', r'$\widehat{\sigma}_{12}$', r'$\widehat{\sigma}_{13}$']

        self.imagesc([tau['uu'][:, :, 127], tau['uv'][:, :, 127], tau['uw'][:, :, 127]],
                     map_bounds, name=name, titles=titles)

    def sigma_pdf(self):

        name = 'sigma_pdf'
        fig, axarr = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(6.5, 2.4))
        titles = [r'$\sigma_{11}$', r'$\sigma_{12}$', r'$\sigma_{13}$']
        tau = g.LES.tau_true
        for ind, i in enumerate(['uu', 'uv', 'uw']):
            data = tau[i].flatten()
            x, y = utils.pdf_from_array_with_x(data, g.bins, g.domain)
            axarr[ind].plot(x, y, 'r', linewidth=2, label='LES')
            axarr[ind].set_xlabel(titles[ind])
        tau = g.TEST.tau_true
        for ind, i in enumerate(['uu', 'uv', 'uw']):
            data = tau[i].flatten()
            x, y = utils.pdf_from_array_with_x(data, g.bins, g.domain)
            axarr[ind].plot(x, y, 'g', linewidth=2, label='test')

        axarr[0].axis(xmin=-1.1, xmax=1.1, ymin=1e-5)
        axarr[0].set_ylabel('pdf')
        axarr[0].set_yscale('log', nonposy='clip')
        plt.legend(loc=0)
        fig.subplots_adjust(left=0.1, right=0.95, wspace=0.1, bottom=0.2, top=0.9)
        fig.savefig(self.folder+name)
        del fig, axarr
        gc.collect()

    def spectra(self):
        os.chdir(self.folder)
        files = glob.glob("*.spectra")
        for k in range(len(files)):
            file = files[k]
            f = open(file, 'r')
            label = file[:3]
            data = np.array(f.readlines()).astype(np.float)
            x = np.arange(len(data))
            plt.loglog(x, data, '-', linewidth=3, label=label)
            y = 7.2e14 * np.power(x, -5 / 3)
            plt.loglog(x, y, 'r--')
        plt.title('Spectrum', fontsize=20)
        plt.ylabel(r'$E$', fontsize=20)
        plt.xlabel(r'k', fontsize=20)
        plt.axis(ymin=1e6)
        plt.legend(loc=0)
        plt.show()

def T_TEST(T_TEST):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(10, 5))
    titles = [r'$T_{11}$', r'$T_{12}$', r'$T_{13}$']
    ax1.hist(T_TEST['uu'].flatten(), bins=100, normed=1, alpha=0.4)
    ax2.hist(T_TEST['uv'].flatten(), bins=100, normed=1, alpha=0.4)
    ax3.hist(T_TEST['uw'].flatten(), bins=100, normed=1, alpha=0.4)

    for ind, ax in enumerate([ax1, ax2, ax3]):
        ax.set_xlabel(titles[ind])
    ax1.axis(xmin=-1.1, xmax=1.1, ymin=1e-5)
    ax1.set_ylabel('pdf')
    ax3.set_yscale('log', nonposy='clip')
    fig.tight_layout()
    plt.show()
    del ax1, ax2, ax3, fig
    gc.collect()


def TS(TS):
    plt.figure(figsize=(5, 5))
    plt.hist(TS.flatten(), bins=500, normed=1, alpha=0.4)
    plt.yscale('log', nonposy='clip')
    plt.xlabel(r'$T_{ij}S^T_{ij}$')
    plt.ylabel(r'pdf($T_{ij}S^T_{ij}$)')
    plt.axis(xmin=-5, xmax=4, ymin=1e-5)
    plt.show()
    gc.collect()


def tau_tau_sp(tau, tau_sp):
    fig, axarr = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(12, 8))
    titles = [r'$T_{11}$', r'$T_{12}$', r'$T_{13}$']

    axarr[0, 0].hist(tau['uu'].flatten(), bins=50, normed=1, alpha=0.4)
    axarr[0, 0].set_xlim(xmin=-1.1, xmax=1.1)
    axarr[0, 0].set_ylim(ymin=1e-5)
    axarr[0, 0].set_yscale('log', nonposy='clip')
    axarr[0, 0].set_ylabel(r'pdf of initial')
    axarr[0, 1].hist(tau['uv'].flatten(), bins=50, normed=1, alpha=0.4)
    axarr[0, 2].hist(tau['uw'].flatten(), bins=50, normed=1, alpha=0.4)
    axarr[1, 0].hist(tau_sp['uu'].flatten(), bins=20, normed=1, alpha=0.4)
    axarr[1, 0].set_ylabel('pdf of sparse')
    axarr[1, 1].hist(tau_sp['uv'].flatten(), bins=20, normed=1, alpha=0.4)
    axarr[1, 2].hist(tau_sp['uw'].flatten(), bins=20, normed=1, alpha=0.4)

    for ind, ax in enumerate([axarr[1, 0], axarr[1, 1], axarr[1, 2]]):
        ax.set_xlabel(titles[ind])

    fig.tight_layout()
    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
    plt.show()
    del fig, axarr
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


def S_compare(field, axarr, titles, label, color):
    for ind, i in enumerate(['uu', 'uv', 'uw']):
        data = field[i].flatten()
        x, y = utils.pdf_from_array(data, 100, [-1.1, 1.1])
        axarr[ind].plot(x, y, 'r', linewidth=2, label=label, color=color)
        axarr[ind].set_xlabel(titles[ind])
    axarr[0].axis(xmin=-1.1, xmax=1.1)
    axarr[0].set_ylabel('pdf')
    # axarr[0].set_yscale('log', nonposy='clip')


def A_compare(field, axarr, titles, M, color):
    x = np.linspace(0, 2 * pi - 2 * pi / M, M)
    for ind, i in enumerate(['uu', 'uv', 'uw']):
        data = field[i][int(M / 2), int(M / 2), :]
        print(data.shape, x.shape)
        axarr[ind].plot(x, data, 'r', linewidth=2, label=str(M), color=color)
        axarr[ind].set_xlabel(titles[ind])
    axarr[0].axis(xmin=0, xmax=2 * pi, ymin=-10, ymax=10)






