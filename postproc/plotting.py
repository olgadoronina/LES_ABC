import gc
import glob

import os
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import abc_code.utils

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


########################################################################################################################
#
########################################################################################################################
params_names = [r'$C_1$', r'$C_2$', r'$C_3$', r'$C_4$', r'$C_5$', r'$C_6$', r'$C_7$', r'$C_8$', r'$C_9$']


def plot_marginal_pdf(N_params, output, plot_folder, C_limits):

    max_value = 0.0
    data = dict()
    for i in range(N_params):
        for j in range(N_params):
            if i < j:
                data[str(i)+str(j)] = np.loadtxt(os.path.join(output, 'marginal' + str(i) + str(j)))
                max_value = max(max_value, np.max(data[str(i)+str(j)]))
    max_value = int(max_value)
    cmap = plt.cm.jet  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]  # extract all colors from the .jet map
    cmaplist[0] = 'white' # force the first color entry to be white
    cmap = cmap.from_list('Custom cmap', cmaplist, max_value)
    fig = plt.figure(figsize=(7, 6.5))

    for i in range(N_params):
        for j in range(N_params):
            if i == j:
                data_marg = np.loadtxt(os.path.join(output, 'marginal' + str(i)))
                ax = plt.subplot2grid((N_params, N_params), (i, i))
                ax.plot(data_marg[0], data_marg[1])
                c_final_dist = np.loadtxt(os.path.join(output, 'C_final_dist'))
                # ax.axvline(mean, linestyle='--', color='g', label='mean')
                # ax.axvline(max, linestyle='--', color='r', label='max')
                ax.axvline(c_final_dist[i], linestyle='--', color='g', label='min dist')
                c_final_joint = np.loadtxt(os.path.join(output, 'C_final_joint'))
                if len(c_final_joint) < 4:
                    for C in c_final_joint:
                        ax.axvline(C[i], linestyle='--', color='b', label='joint max')
                ax.axis(xmin=C_limits[i, 0], xmax=C_limits[i, 1], ymin=0)
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
                ax.set_xlabel(params_names[i])
            elif i < j:
                ax = plt.subplot2grid((N_params, N_params), (i, j))
                edges = np.loadtxt(os.path.join(output, 'marginal_bins'+str(i)+str(j)))
                ax.axis(xmin=C_limits[j, 0], xmax=C_limits[j, 1], ymin=C_limits[i, 0], ymax=C_limits[i, 1])
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
                ext = (edges[0, 0], edges[0, -1], edges[1, 0], edges[1, -1])

                im = ax.imshow(data[str(i)+str(j)].T, origin='lower', cmap=cmap, aspect='auto',
                               extent=ext, vmin=0, vmax=max_value)
    cax = plt.axes([0.1, 0.05, 0.075, 0.1])
    plt.colorbar(im, cax=cax) #, ticks=np.arange(max_value+1))
    plt.legend(loc='lower left', bbox_to_anchor=(-6.5, 3.5), fancybox=True, shadow=True)

    if N_params == 3:
        fig.subplots_adjust(left=0.05, right=0.98, wspace=0.25, bottom=0.08, top=0.95)
    elif N_params == 4:
        fig.subplots_adjust(left=0.03, right=0.98, wspace=0.35, hspace=0.3, bottom=0.08, top=0.97)
    elif N_params == 6:
        fig.subplots_adjust(left=0.05, right=0.98, wspace=0.45, hspace=0.35, bottom=0.08, top=0.98)

    fig.savefig(plot_folder+'marginal')
    plt.close('all')


def plot_scatter(N_params, C_limits, visua, accepted, dist):

    for i in range(N_params):
        x = accepted[:, i]
        fig = plt.figure(figsize=(3.2, 2.8))
        ax = plt.axes()
        ax.axis(xmin=C_limits[i, 0], xmax=C_limits[i, 1], ymax=np.max(dist) + 1)
        ax.scatter(x, dist, color='blue')
        ax.set_xlabel(params_names[i])
        ax.set_ylabel(r'$\sum_{i,j}\rho(\mathcal{S}_{ij}^{\mathcal{F}},\mathcal{S}_{ij})$')
        fig.subplots_adjust(left=0.19, right=0.95, bottom=0.15, top=0.9)
        fig.savefig(os.path.join(visua, 'scatter_plot_'+params_names[i][1:-1]))
    plt.close('all')


def plot_compare_tau(visua, output, scale='LES'):

    fig, axarr = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(6.5, 2.5))
    titles = [r'$\sigma_{11}$', r'$\sigma_{12}$', r'$\sigma_{13}$']
    x = np.loadtxt(os.path.join(output, 'sum_stat_bins'))
    y_true = dict()
    y_min_dist = dict()
    for ind, key in enumerate(['uu', 'uv', 'uw']):
        # Plot true pdf
        y_true[key] = np.loadtxt(os.path.join(output, 'sum_stat_true'))[ind]
        axarr[ind].plot(x, y_true[key], 'r', linewidth=2, label='true')
        # axarr[ind].xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        # plot min dist pdf
        y_min_dist[key] = np.loadtxt(os.path.join(output, 'sum_stat_min_dist_' + scale))[ind]
        axarr[ind].plot(x, y_min_dist[key], 'g', linewidth=2, label='modeled dist')
        # # plot max marginal
        # x, y = utils.pdf_from_array_with_x(tau_modeled_marginal[key].flatten(), g.bins, g.domain)
        # axarr[ind].plot(x, y, 'm', linewidth=2, label='modeled marginal max')
        axarr[ind].set_xlabel(titles[ind])

    # # Plot max joint pdf
    # if C_final_joint:
    #     for i in range(len(C_final_joint)):
    #         tau_modeled_joint = current_model.Reynolds_stresses_from_C_tau(C_final_joint[i])
    #         y_dict = dict()
    #         for ind, key in enumerate(['uu', 'uv', 'uw']):
    #             x, y_dict[key] = utils.pdf_from_array_with_x(tau_modeled_joint[key].flatten(), g.bins, g.domain)
    #             y = utils.take_safe_log(y_dict[key])
    #             axarr[ind].plot(x, y, 'b', linewidth=2, label='modeled joint')
    #
    #         np.savez('./plots/pdf.npz', x=x, uu=y_dict['uu'], uv=y_dict['uv'], uw=y_dict['uw'])
    #
    # axarr[0].axis(xmin=self.domain[0], xmax=self.domain[1], ymin=-7)      #ymin=g.TINY_log-0.5)

    axarr[0].set_ylabel('ln(pdf)')
    plt.legend(loc=0)
    fig.subplots_adjust(left=0.1, right=0.95, wspace=0.1, bottom=0.18, top=0.9)
    # axarr[0].set_yscale('log', basey=np.e)
    fig.savefig(os.path.join(visua, 'compare_sum_stat_' + scale))
    plt.close('all')

# def plot_eps(self):
#     num_eps = 6
#     eps = np.linspace(200, 4000, num_eps)
#
#     # eps = np.append(8.877, eps)
#
#     C_mean = np.empty((self.N.params, num_eps))
#     C_max = np.empty_like(C_mean)
#     C_std = np.empty((self.N.params, num_eps))
#     C_h = np.empty((self.N.params, num_eps))
#
#     fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(6.5, 2.5))
#     for ind, new_eps in enumerate(eps):
#         g.accepted = np.load('./plots/accepted.npz')['C']
#         g.dist = np.load('./plots/accepted.npz')['dist']
#
#         g.accepted = g.accepted[g.dist < new_eps]
#         g.dist = g.dist[g.dist < new_eps]
#         logging.info('eps = {}: accepted {} values ({}%)'.format(new_eps,
#             len(g.accepted), round(len(g.accepted) / (g.N.total) * 100, 2)))
#         for i in range(self.N.params):
#             data = g.accepted[:, i]
#             C_std[i, ind] = np.std(data)
#             C_mean[i, ind], C_h[i, ind] = utils.mean_confidence_interval(data, confidence=0.95)
#
#             x, y = utils.pdf_from_array_with_x(data, bins=self.N.each, range=self.C_limits[i])
#             C_max[i, ind] = x[np.argmax(y)]
#             axarr[i].plot(x, y, label=r'$\epsilon = {}$'.format(new_eps))
#             axarr[i].set_xlabel(self.params_names[i])
#
#     axarr[0].set_ylabel('marginal pdf')
#     # Put a legend below current axis
#     legend = plt.legend(loc='upper center', bbox_to_anchor=(1., 1.1))
#     # frame = legend.get_frame()
#     # frame.set_alpha(1)
#     fig.subplots_adjust(left=0.08, right=0.9, wspace=0.17, bottom=0.17, top=0.9)
#     fig.savefig(self.folder + 'eps_marginal')
#
#     fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(6.5, 2.5))
#     for i in range(self.N.params):
#         axarr[i].plot(eps, C_mean[i], 'b.-', label='mean')
#         axarr[i].plot(eps, C_max[i], 'g.-', label='max')
#         axarr[i].set_title(self.params_names[i])
#         axarr[i].set_xlabel('epsilon')
#         axarr[i].xaxis.set_major_locator(ticker.MultipleLocator(5))
#     axarr[0].set_ylabel(r'$C_i$')
#     plt.legend(loc=0)
#     fig.subplots_adjust(left=0.1, right=0.97, wspace=0.4, bottom=0.2, top=0.85)
#     fig.savefig(self.folder + 'eps_plot')
#
#
#
#     fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(6.5, 2.5))
#     for i in range(self.N.params):
#         axarr[i].plot(eps, C_std[i], 'b.-')
#         axarr[i].set_title(self.params_names[i])
#         axarr[i].set_xlabel('epsilon')
#         axarr[i].xaxis.set_major_locator(ticker.MultipleLocator(5))
#         # axarr[i].axis(ymin=np.min(C_mean[i])-0.01*, ymax=np.max(C_mean[i])+0.1)
#     axarr[0].set_ylabel(r'std($C_i$)')
#     fig.subplots_adjust(left=0.1, right=0.97, wspace=0.4, bottom=0.2, top=0.85)
#     fig.savefig(self.folder + 'eps_std')
#
#
#     fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(6.5, 2.5))
#     for i in range(self.N.params):
#         axarr[i].plot(eps, C_h[i], 'b.-')
#         axarr[i].set_title(self.params_names[i])
#         axarr[i].set_xlabel('epsilon')
#         axarr[i].xaxis.set_major_locator(ticker.MultipleLocator(5))
#         # axarr[i].axis(ymin=np.min(C_mean[i])-0.01*, ymax=np.max(C_mean[i])+0.1)
#     axarr[0].set_ylabel(r'$95\%$ confident interval')
#     fig.subplots_adjust(left=0.12, right=0.97, wspace=0.4, bottom=0.2, top=0.85)
#     fig.savefig(self.folder + 'eps_h')