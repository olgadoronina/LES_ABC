import gc
import glob

import os
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import utils

# mpl.style.use(['dark_background','mystyle'])
# mpl.style.use(['mystyle'])

# mpl.rcParams['figure.figsize'] = 6.5, 2.2
# plt.rcParams['figure.autolayout'] = True

mpl.rcParams['font.size'] = 10
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rc('text', usetex=True)
mpl.rcParams['axes.labelsize'] = plt.rcParams['font.size']
mpl.rcParams['axes.titlesize'] = 1.5 * plt.rcParams['font.size']
mpl.rcParams['legend.fontsize'] = plt.rcParams['font.size']
mpl.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
mpl.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
# plt.rcParams['savefig.dpi'] = 2 * plt.rcParams['savefig.dpi']
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.width'] = 1
# mpl.rcParams['legend.frameon'] = False
# plt.rcParams['legend.loc'] = 'center left'
plt.rcParams['axes.linewidth'] = 1


def imagesc(Arrays, map_bounds, titles, name=None):

    cmap = plt.cm.jet  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]  # extract all colors from the .jet map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)  # create the new map
    norm = mpl.colors.BoundaryNorm(map_bounds, cmap.N)

    axis = [0, 2 * np.pi, 0, 2 * np.pi]
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
        fig.savefig(name)
    plt.close('all')

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


def compare_filter_fields(hit_data, les_data, test_data, map_bounds, folder):

    name = os.path.join(folder, 'compare_velocity')
    if test_data:
        imagesc([hit_data['v'][:, :, 127], les_data['v'][:, :, 127], test_data['v'][:, :, 127]],
                map_bounds=map_bounds, name=name,
                titles=[r'$v$', r'$\widetilde{v}$', r'$\widehat{\widetilde{v}}$'])
    else:
        imagesc([hit_data['v'][:, :, 127], les_data['v'][:, :, 127]],
                map_bounds=map_bounds, name=name,
                titles=[r'$v$', r'$\widetilde{v}$'])


def vel_fields(data, scale, map_bounds, folder):
    """ Plot velocity fields"""
    if scale == 'DNS':
        titles = [r'$u$', r'$v$', r'$w$']
        name = os.path.join(folder, 'DNS_velocities')
    elif scale == 'LES':
        titles = [r'$\widetilde{u}$', r'$\widetilde{v}$', r'$\widetilde{w}$']
        name = os.path.join(folder, 'LES_velocities')
    elif scale == 'TEST':
        titles = [r'$\widehat{\widetilde{u}}$', r'$\widehat{\widetilde{v}}$', r'$\widehat{\widetilde{w}}$']
        name = os.path.join(folder, 'TEST_velocities')
    imagesc([data['u'][:, :, 127], data['v'][:, :, 127], data['w'][:, :, 127]],
            name=name, titles=titles, map_bounds=map_bounds)


def sigma_field(tau, scale, folder):

    map_bounds = np.linspace(-0.2, 0.2, 9)
    if scale == 'LES':
        name = os.path.join(folder, 'sigma_LES')
        titles = [r'$\sigma_{11}$', r'$\sigma_{12}$', r'$\sigma_{13}$']

    elif scale == 'TEST':
        name = os.path.join(folder, 'sigma_TEST')
        titles = [r'$\widehat{\sigma}_{11}$', r'$\widehat{\sigma}_{12}$', r'$\widehat{\sigma}_{13}$']

    imagesc([tau['uu'][:, :, 127], tau['uv'][:, :, 127], tau['uw'][:, :, 127]],
            map_bounds, name=name, titles=titles)


def sum_stat(les, test, bins, domain, folder, name):

    labels = ['LES', 'test']

    if name == 'sigma_pdf_log':
        fig, axarr = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(6.5, 2.4))
        if test:
            titles = [r'$\sigma_{11},\ \widehat{\sigma}_{11}$',
                    r'$\sigma_{12},\ \widehat{\sigma}_{12}$',
                    r'$\sigma_{13},\ \widehat{\sigma}_{13}$']
        else:
            titles = [r'$\sigma_{11}$', r'$\sigma_{12}$', r'$\sigma_{13}$']

        for ind, i in enumerate(['uu', 'uv', 'uw']):
            data = les.sum_stat_true[i].flatten()
            x, y = utils.pdf_from_array_with_x(data, bins, domain)
            axarr[ind].plot(x, y, 'r', linewidth=2, label=labels[0])
            axarr[ind].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        if test:
            for ind, i in enumerate(['uu', 'uv', 'uw']):
                data = test.sum_stat_true[i].flatten()
                x, y = utils.pdf_from_array_with_x(data, bins, domain)
                axarr[ind].plot(x, y, 'g', linewidth=2, label=labels[1])
                axarr[ind].set_xlabel(titles[ind])

        axarr[0].axis(xmin=-1.1, xmax=1.1, ymin=1e-5)
        axarr[0].set_ylabel('pdf')
        axarr[0].set_yscale('log')
        plt.legend(loc=0)
        fig.subplots_adjust(left=0.1, right=0.95, wspace=0.1, bottom=0.2, top=0.9)

    elif name == 'production_pdf_log':
        fig = plt.figure(figsize=(4, 3))
        ax = plt.gca()
        if test:
            title = r'$\widehat{\widetilde{P}}'
        else:
            title = r'$\widetilde{P}'

        data = les.sum_stat_true.flatten()
        x, y = utils.pdf_from_array_with_x(data, bins, domain)
        ax.plot(x, y, 'r', linewidth=2, label=labels[0])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        if test:
            data = test.sum_stat_true.flatten()
            x, y = utils.pdf_from_array_with_x(data, bins, domain)
            ax.plot(x, y, 'g', linewidth=2, label=labels[1])
            ax.set_xlabel(title)

        ax.axis(xmin=-5, xmax=5, ymin=1e-5)
        ax.set_ylabel('pdf')
        ax.set_yscale('log')
        plt.legend(loc=0)
        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)
    fig.savefig(os.path.join(folder, name))
    plt.close('all')


def S_pdf(folder):

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
    fig.savefig(folder + name)
    del fig, axarr
    gc.collect()



def spectra(folder):

    fig = plt.figure(figsize=(4, 3))
    ax = plt.gca()
    files = glob.glob(folder + '*.spectra')

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
    fig.savefig(folder + 'spectra')

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


def dist_pdf(dist, x, folder):

    fig = plt.figure(figsize=(5, 3))
    ax = plt.gca()
    ax.hist(dist, bins=100, normed=1, alpha=0.4)
    eps = np.percentile(dist, q=int(x * 100))
    ax.axvline(eps, label='eps')
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel(r'pdf($\rho$)')
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)
    fig.savefig(os.path.join(folder, 'dist'))
    plt.close('all')




