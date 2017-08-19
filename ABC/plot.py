import calculate
from params import *
import global_var as g

def imagesc(Arrays, map_bounds, name=None, titles=None, N_proc=None):
    cmap = plt.cm.jet  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]  # extract all colors from the .jet map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)  # create the new map
    norm = mpl.colors.BoundaryNorm(map_bounds, cmap.N)

    # fig = plt.figure()
    if N_proc:
        axis = [0, 2 * pi/N_proc, 0, 2 * pi]
    else:
        axis = [0, 2 * pi, 0, 2 * pi]
    if not titles:
        titles = ['DNS data', 'LES filter', 'Test filter']
    if len(Arrays) > 1:
        fig, axes = plt.subplots(nrows=1, ncols=len(Arrays), figsize=(15, 4))
        k = 0
        for ax in axes.flat:
            im = ax.imshow(Arrays[k].T, origin='lower', cmap=cmap, norm=norm, interpolation="nearest", extent=axis)
            ax.set_title(titles[k])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            k += 1
        cbar_ax = fig.add_axes([0.87, 0.10, 0.017, 0.80])  # ([0.85, 0.15, 0.05, 0.68])
        fig.subplots_adjust(right=0.85)
        fig.colorbar(im, cax=cbar_ax, ax=axes.ravel().tolist())
    else:
        fig = plt.figure(figsize=(12, 10))
        ax = plt.gca()
        im = ax.imshow(Arrays[0].T, origin='lower', cmap=cmap, norm=norm, interpolation="nearest")
        fig.tight_layout()
        plt.colorbar(im, fraction=0.05, pad=0.04)

    fig1 = plt.gcf()
    # plt.show()
    if name:
        fig1.savefig('./plots/'+ name + '.eps')
    del ax, im, fig, fig1, cmap
    gc.collect()


def histogram(field, bins, pdf=None, label=None, log=False):
    plt.figure(figsize=(6, 4))
    plt.hist(field, bins=bins, alpha=0.4, normed=1)

    # h, edges = np.histogram(field, bins=bins, range=[-2,2], normed=1)
    if pdf:
        x = np.linspace(min(field), max(field), 100)
        mu = np.mean(field)
        sigma = np.std(field)
        plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r--', linewidth=3, label='Gaussian')
        plt.legend(loc=0)
    if log:
        plt.yscale('log', nonposy='clip')
    if label:
        plt.xlabel(label)
        plt.ylabel('pdf(' + label + ')')
    plt.axis(xmin=np.min(field), xmax=np.max(field))  # xmax=4xmax = np.max(field)
    plt.show()
    gc.collect()


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

def tau_sp(tau_sp, name=None):
    fig, axarr = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(12, 4))
    titles = [r'$\widetilde{\tau}_{11}$', r'$\widetilde{\tau}_{12}$', r'$\widetilde{\tau}_{13}$']
    for ind, i in enumerate(['uu', 'uv', 'uw']):
        data = tau_sp[i].flatten()
        x, y = utils.pdf_from_array(data, 100, [-1.1, 1.1])
        axarr[ind].plot(x, y, 'r', linewidth=2)
        axarr[ind].set_xlabel(titles[ind])
    axarr[0].axis(xmin=-1.1, xmax=1.1, ymin=1e-5)
    axarr[0].set_ylabel('pdf')
    axarr[0].set_yscale('log', nonposy='clip')
    # fig.tight_layout()
    fig1 = plt.gcf()
    plt.show()
    if name:
        fig1.savefig(name + '.eps')
    del fig, fig1, axarr
    gc.collect()

def tau_abc(Cs_abc):
    fig, axarr = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(18, 6))
    titles = [r'$\widehat{T}_{11}$', r'$\widehat{T}_{12}$', r'$\widehat{T}_{13}$']

    x = np.linspace(domain[0], domain[1], bins)
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
            x, y = utils.pdf_from_array(tau[i].flatten(), bins, domain)
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
        axarr[i].set_position([box.x0, box.y0, box.width*0.9, box.height])
    # Put a legend below current axis
    plt.legend(loc='upper center', bbox_to_anchor=(1.6, 1.1),  fancybox=True, shadow=True, ncol=2)
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig('tau_abc.eps')
    del fig, axarr
    gc.collect()


def C_scatter(Cs_accepted, dist,  Cs_failed = None, label=None):

    if Cs_failed is not None:
        plt.scatter(Cs_failed[:, 0], Cs_failed[:, 1], color='red')
        plt.axhline(y=eps, color='r', linestyle='--')
        # plt.axis(xmin=C_limits[0, 0], xmax=C_limits[0, 1])
    # else:
    if label == r'$C_s$':
        plt.axis(xmin=0.18, xmax=0.3, ymax=50)
    elif label == r'$C_2$':
        plt.axis(xmin=C_limits[1, 0], xmax=C_limits[1, 1], ymax=75)
    elif label == r'$C_4$':
        plt.axis(xmin=C_limits[3, 0], xmax=C_limits[3, 1], ymax=75)
    else:
        plt.axis(xmin=C_limits[2, 0], xmax=C_limits[2, 1], ymax=75)
    plt.scatter(Cs_accepted, dist, color='blue')
    if label==None:
        plt.xlabel(r'$C_s$')
    plt.xlabel(label)
    plt.ylabel(r'$\sum_{i,j}\rho(\widehat{T}_{ij}^{\mathcal{F}},\widehat{T}_{ij})$')
    plt.show()
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

    x = np.linspace(0, 2*pi-2*pi/M, M)
    for ind, i in enumerate(['uu', 'uv', 'uw']):
        data = field[i][int(M/2), int(M/2), :]
        print(data.shape, x.shape)
        axarr[ind].plot(x, data, 'r', linewidth=2, label=str(M), color=color)
        axarr[ind].set_xlabel(titles[ind])
    axarr[0].axis(xmin=0, xmax=2*pi, ymin=-10, ymax=10)


def plot_LES_vel_fields():
    if not g.LES:
        logging.warning('Can not plot LES field: g.LES is None')
    map_bounds = np.linspace(np.min(g.LES.field['u'][:, :, 127]), np.max(g.LES.field['v'][:, :, 127]), 20)
    imagesc([g.LES.field['u'][:, :, 127], g.LES.field['v'][:, :, 127], g.LES.field['w'][:, :, 127]], map_bounds,
                 name='LES_velocities', titles=[r'$\widetilde{u}$', r'$\widetilde{v}$', r'$\widetilde{w}$'])

def plot_compare_filtered_fields():
    if not g.HIT or not g.LES or not g.TEST:
        logging.warning('Can not plot fields: some of them is None')
    map_bounds = np.linspace(np.min(g.HIT.field['u'][:, :, 127]), np.max(g.HIT.field['u'][:, :, 127]), 20)
    imagesc([g.HIT.field['u'][:, :, 127], g.LES.field['u'][:, :, 127], g.TEST.field['u'][:, :, 127]], map_bounds, 'fourier_tophat')

def plot_LES_tau():
    if not g.LES:
        logging.warning('Can not plot LES field: g.LES is None')
    map_bounds = np.linspace(-0.2, 0.2, 10)
    imagesc([g.LES.tau_true['uu'][:, :, 127], g.LES.tau_true['uv'][:, :, 127], g.LES.tau_true['uw'][:, :, 127]], map_bounds,
                 name='tau_LES', titles=[r'$\widetilde{\tau_{11}}$', r'$\widetilde{\tau_{12}}$', r'$\widetilde{\tau_{13}}$'])