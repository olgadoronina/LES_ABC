from params import *
import random as rand
import calculate
import utils
import plot
import multiprocessing as mp

def distance_between_pdf(pdf_modeled, pdf_true):
    """Calculate statistical distance between two pdf as
    the Kullback-Leibler (KL) divergence (no symmetry).
    In the simple case, a KL divergence of 0 indicates that we can expect similar,
    while a KLr divergence of 1 indicates that the two distributions behave in a different manner.
    :param pdf_true:    array of expected pdf
    :param pdf_modeled: array of modeled pdf
    :return:            scalar of calculated distance
    """
    log_true = np.log(pdf_true, out=np.zeros_like(pdf_true), where=pdf_true != 0)
    if np.isnan(np.sum(log_true)):
        print(log_true)
    log_modeled = np.log(pdf_modeled, out=np.zeros_like(pdf_modeled), where=pdf_modeled != 0)
    if np.isnan(np.sum(log_modeled)):
        print(log_modeled)
    dist = np.sum(np.multiply(pdf_modeled, (log_modeled - log_true)))
    return dist

def work_function(Cs_array, test_field, pdf_true, S_mod_S_ij):
    result = [[], []]
    for C_s in Cs_array:
        tau = calculate.Reynolds_stresses_from_Cs(test_field, C_s, TEST_delta, S_mod_S_ij=S_mod_S_ij)
        print('C_s = ', C_s)
        dist = 0
        for key, value in tau.items():
            x, y = utils.pdf_from_array(value.flatten(), bins, domain)
            dist += distance_between_pdf(pdf_modeled=y, pdf_true=pdf_true[key])
        if dist <= eps:
            result[0].append(C_s)
            result[1].append(dist)
        print(dist)
    return result


def ABC(tau_exact, test_field):

    x = np.linspace(domain[0], domain[1], bins)
    pdf_true = dict()
    for key, value in tau_exact.items():
        pdf_true[key] = utils.pdf_from_array(value, bins, domain)[1]

    S_mod_S_ij = calculate.strain_mod_strain_ij(test_field)

    fig, axarr = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(15, 6))
    titles = [r'$T_{11}$', r'$T_{12}$', r'$T_{13}$']

    for ind, i in enumerate(['uu', 'vu', 'wu']):
        axarr[ind].plot(x, pdf_true[key], linewidth=3, label='true pdf')
        axarr[ind].set_xlabel(titles[ind])

    # tau = calculate.Reynolds_stresses_from_Cs(test_field, 0.2, TEST_delta)
    # for ind, i in enumerate(['uu', 'vu', 'wu']):
    #     x, y = utils.pdf_from_array(tau[i].flatten(), bins, domain)
    #     axarr[ind].plot(x, y, linewidth=3, label='Cs= 0.2')

    # tau = calculate.Reynolds_stresses_from_Cs(test_field, 0.0685, TEST_delta)
    # for ind, i in enumerate(['uu', 'vu', 'wu']):
    #     x, y = utils.pdf_from_array(tau[i].flatten(), bins, domain)
    #     axarr[ind].plot(x, y, linewidth=3, label='Cs= 0.0685')

    C_s_array = [rand.uniform(0, 0.4) for i in range(N)]

    # Main loop
    ####################################################################################################################
    start = time()
    if PARALLEL:
        p = mp.Pool(mp.cpu_count())
        result = p.map(work_function(C_s_array, test_field, pdf_true, S_mod_S_ij), C_s_array)  # in parallel
        p.close()
        p.join()
    else:
        result = work_function(C_s_array, test_field, pdf_true, S_mod_S_ij)
    end = time()
    utils.timer(start, end, 'Time ')
    ####################################################################################################################
    for C_s in result[0]:
        tau = calculate.Reynolds_stresses_from_Cs(test_field, C_s, TEST_delta, S_mod_S_ij)
        for ind, i in enumerate(['uu', 'vu', 'wu']):
            x, y = utils.pdf_from_array(tau[i].flatten(), bins, domain)
            axarr[ind].plot(x, y, linewidth=1, label='Cs= ' + str(round(C_s, 2)))
    axarr[0].axis(xmin=-1.1, xmax=1.1, ymin=1e-5)
    axarr[0].set_ylabel('pdf')
    axarr[0].set_yscale('log', nonposy='clip')
    fig.tight_layout()
    plt.legend(loc=0)
    plt.show()

    plt.scatter(result[0], result[1])
    plt.xlabel(r'$C_s$')
    plt.ylabel(r'$\rho( \hat{T}_{ij},T_{ij})$')
    plt.show()

