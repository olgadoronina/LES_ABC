import params
import global_var as g
import init

import numpy as np
import logging
import matplotlib.pyplot as plt
# import matplotlib as mpl

filename = './plots/sweep_params.npz'
filename_h = './plots/sweep_h.npz'
folder = './plots/'


# mpl.rcParams['font.size'] = 10
# mpl.rcParams['font.family'] = 'Times New Roman'
# mpl.rc('text', usetex=True)
# mpl.rcParams['axes.labelsize'] = mpl.rcParams['font.size']
# mpl.rcParams['axes.titlesize'] = 1.5 * mpl.rcParams['font.size']
# mpl.rcParams['legend.fontsize'] = mpl.rcParams['font.size']
# mpl.rcParams['xtick.labelsize'] = mpl.rcParams['font.size']
# mpl.rcParams['ytick.labelsize'] = mpl.rcParams['font.size']
# # plt.rcParams['savefig.dpi'] = 2 * plt.rcParams['savefig.dpi']
# mpl.rcParams['xtick.major.size'] = 3
# mpl.rcParams['xtick.minor.size'] = 3
# mpl.rcParams['xtick.major.width'] = 1
# mpl.rcParams['xtick.minor.width'] = 1
# mpl.rcParams['ytick.major.size'] = 3
# mpl.rcParams['ytick.minor.size'] = 3
# mpl.rcParams['ytick.major.width'] = 1
# mpl.rcParams['ytick.minor.width'] = 1
# # mpl.rcParams['legend.frameon'] = False
# # plt.rcParams['legend.loc'] = 'center left'
# mpl.rcParams['axes.linewidth'] = 1


def nominal_sweep(N):

    C_array = []
    C_nominal = [0.182, -0.0525,  0.0196, 0.0215,  0.4035,  0.0106, 0, 0, 0, 0]
    C_nominal[0] = -2*C_nominal[0]**2
    C_nominal = normalize_params(C_nominal)
    c_sweep = np.linspace(-1, 1, N.each)
    for i in range(N.params):
        for c_j in c_sweep:
            c = C_nominal.copy()
            c[i] = c_j
            c = unnormalize_params(c)
            C_array.append(list(c))
    return C_array

def random_sweep(N):
    """ Create list of lists of N parameters.
        :return: list of lists of sampled parameters
        """
    C_array = []
    h_array = []
    for i in range(params.n_sweeps):
        # 1. Draw C uniformly from [-1; 1]^m
        x = np.random.uniform(-1, 1, size=N.params)
        # 2. pick a random direction
        u = np.random.normal(0, 1, size=N.params)
        u = u / np.linalg.norm(u)  # unit vector
        # 3. Compute h_min and h_max
        ind_pos = np.where(u >= 0)
        ind_neg = np.where(u < 0)
        h_min = np.max(np.append((1 - x[ind_neg]) / u[ind_neg], (-1 - x[ind_pos]) / u[ind_pos]))
        h_max = np.min(np.append((1 - x[ind_pos]) / u[ind_pos], (-1 - x[ind_neg]) / u[ind_neg]))
        delta_h = (h_max - h_min) / N.each
        # 4. Compute parameters points
        for j in range(N.each):
            h = h_min + j * delta_h
            h_array.append(h + x.dot(u))
            C_array.append(list(unnormalize_params(x + h * u)))

    np.savez('./plots/sweep_h.npz', h=h_array[:])
    logging.info('Accepted parameters and distances saved in ./ABC/plots/sweep_h.npz')

    return list(C_array)


def normalize_params(C):
    """
    Normalize given parameters to -1<C<1 hypercube.
    :param C: given parameters
    :return: normalized parameters
    """
    return 2*(C - g.C_limits[:, 0])/(g.C_limits[:, 1] - g.C_limits[:, 0]) - 1


def unnormalize_params(C):
    return 0.5*(C+1)*(g.C_limits[:, 1] - g.C_limits[:, 0]) + g.C_limits[:, 0]

########################################################################################################################
# plotting routine
########################################################################################################################
def plot_sweep_nominal(N):

    dist = np.load(filename)['dist']
    labels = [r'$C_S$', r'$C_2$', r'$C_3$', r'$C_4$', r'$C_5$', r'$C_6$', r'$C_7$', r'$C_8$', r'$C_9$', r'$C_{10}$']
    n_rows = 3
    n_columns = 2
    print(n_rows, n_columns)
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes()
    x = np.linspace(-1, 1, N.each)
    for i in range(N.params):
        ax.axis(xmin=-1, xmax=1, ymax=15)
        ax.plot(x, dist[i*N.each:(i+1)*N.each], label=labels[i], linewidth=2)
        ax.set_xlabel(r'$C_i$')
        ax.set_ylabel(r'$\sum_{i,j}\rho(\mathcal{S}_{ij}^{\mathcal{F}},\mathcal{S}_{ij})$')
        ax.set_title('Parameters sweep')
    plt.legend(loc=0)
    fig.subplots_adjust(left=0.17, right=0.95, bottom=0.1, top=0.9, hspace=0.15)

    fig.savefig(folder + 'sweep_zoom')
    del fig


def plot_sweep_random(N):

    dist = np.load(filename)['dist']
    h = np.load(filename_h)['h']
    print(dist.shape, h.shape)
    print(np.max(dist), np.argmax(dist))
    n_rows = 4
    n_columns = 4
    print(n_rows, n_columns)
    fig = plt.figure(figsize=(9, 9))
    plt.title('Random parameters sweep')
    for i in range(16):
        ax = plt.subplot2grid((n_rows, n_columns), (i // n_rows, i % n_rows))
        ax.axis(xmin=np.min(h), xmax=np.max(h), ymax=np.max(dist))
        for j in range(5):
            start = (i*5+j)*N.each
            end = (i*5+j+1)*N.each
            print(i, start, end)
            ax.plot(h[start:end], dist[start:end])
        if i % n_rows == 0:
            ax.set_ylabel(r'$\sum_{i,j}\rho(\mathcal{S}_{ij}^{\mathcal{F}},\mathcal{S}_{ij})$')

    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, hspace=0.15)
    fig.savefig(folder + 'random_sweep')
    del fig


def main():
    logging.basicConfig(format='%(levelname)s: %(name)s: %(message)s', level=logging.DEBUG)
    N = init.NPoints()
    plot_sweep_random(N)

if __name__ == '__main__':
    main()