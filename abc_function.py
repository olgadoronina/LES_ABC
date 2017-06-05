from params import *
import random as rand
import calculate
import plot
from scipy.stats.kde import gaussian_kde

def pdf_from_array(array, bins, range):
    pdf, edges = np.histogram(array, bins=bins, range=range, normed=1)
    x = (edges[1:] + edges[:-1])/2
    return x, pdf

def distance_between_pdf(pdf_true, pdf_modeled):
    """Calculate statistical distance between two pdf as
    the Kullback-Leibler (KL) divergence (no symmetry).
    In the simple case, a KL divergence of 0 indicates that we can expect similar,
    while a KLr divergence of 1 indicates that the two distributions behave in a different manner.
    :param pdf_true:    array of expected pdf
    :param pdf_modeled: array of modeled pdf
    :return:            scalar of calculated distance
    """
    log_true = np.log(pdf_true)
    log_modeled = np.log(pdf_modeled)
    dist = np.multiply(pdf_modeled, (log_modeled - log_true))
    return dist


def ABC(tau_exact, test_field, N):
    C_s_array = [rand.random() for i in range(N)]

    fig, axarr = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(15, 6))
    titles = [r'$T_{11}$', r'$T_{12}$', r'$T_{13}$']

    tau = calculate.Reynolds_stresses_from_Cs(test_field, 0.2)
    for ind, i in enumerate(['uu', 'vu', 'wu']):
        data = tau[i].flatten()
        kde = gaussian_kde(data)
        dist_space = np.linspace(min(data), max(data), 100)
        # axarr[ind].hist(tau_sp[i].flatten(), bins=50, normed=1, alpha=0.4)
        axarr[ind].plot(dist_space, kde(dist_space), linewidth=3, label=r'C_s = 0.2')
        axarr[ind].set_xlabel(titles[ind])

    tau = calculate.Reynolds_stresses_from_Cs(test_field, 0.2)
    for ind, i in enumerate(['uu', 'vu', 'wu']):
        data = tau[i].flatten()
        kde = gaussian_kde(data)
        dist_space = np.linspace(min(data), max(data), 100)
        # axarr[ind].hist(tau_sp[i].flatten(), bins=50, normed=1, alpha=0.4)
        axarr[ind].plot(dist_space, kde(dist_space), linewidth=3, label=r'$C_s$= 0.01')
        axarr[ind].set_xlabel(titles[ind])

    for C_s in C_s_array:
        tau = calculate.Reynolds_stresses_from_Cs(test_field, C_s)
        print('C_s = ', C_s)
        print(tau['uu'].shape, tau['uv'].shape, tau['uw'].shape)
        for ind, i in enumerate(['uu', 'vu', 'wu']):
            data = tau[i].flatten()
            kde = gaussian_kde(data)
            dist_space = np.linspace(min(data), max(data), 100)
            # axarr[ind].hist(tau_sp[i].flatten(), bins=50, normed=1, alpha=0.4)
            axarr[ind].plot(dist_space, kde(dist_space), linewidth=1, label=r'$C_s$ = '+str(round(C_s,2)))
            axarr[ind].set_xlabel(titles[ind])
    axarr[0].axis(xmin=-400, xmax=400, ymin=1e-5)
    axarr[0].set_ylabel('pdf')
    axarr[0].set_yscale('log', nonposy='clip')
    fig.tight_layout()
    plt.legend(loc=0)
    plt.show()