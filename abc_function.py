from params import *
import random as rand
import calculate
import utils
import plot
import multiprocessing as mp
from functools import partial

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


def work_function(Cs, test_field, pdf_true, S_mod_S_ij):
    tau = calculate.Reynolds_stresses_from_Cs(test_field, Cs, TEST_delta, S_mod_S_ij=S_mod_S_ij)
    print('C_s = ', Cs)
    dist = 0
    for key, value in tau.items():
        x, y = utils.pdf_from_array(value.flatten(), bins, domain)
        dist += distance_between_pdf(pdf_modeled=y, pdf_true=pdf_true[key])
    print(dist)
    if dist <= eps:
        return [Cs, dist]


def ABC(tau_exact, test_field):

    pdf_true = dict()
    for key, value in tau_exact.items():
        pdf_true[key] = utils.pdf_from_array(value, bins, domain)[1]
    S_mod_S_ij = calculate.strain_mod_strain_ij(test_field)

    Cs_array = [rand.uniform(Cs_limits[0], Cs_limits[1]) for i in range(N)]
    # Main loop
    ####################################################################################################################
    start = time()

    if PARALLEL:
        print("\n%d workers" % mp.cpu_count())
        pool = mp.Pool(mp.cpu_count())
        func = partial(work_function, test_field=test_field, pdf_true=pdf_true, S_mod_S_ij=S_mod_S_ij)
        result = pool.map(func, Cs_array)  # in parallel
        pool.close()
        pool.join()
        print(result)
    else:
        result = [[], []]
        for Cs in Cs_array:
            work_function(Cs, test_field, pdf_true, S_mod_S_ij)
    end = time()
    utils.timer(start, end, 'Time ')
    print(result)
    ####################################################################################################################

    plot.tau_abc(pdf_true, result[0], test_field, S_mod_S_ij)
    plot.Cs_scatter(result[0], result[1])

    Cs_final = result[0][result[1].index(min(result[1]))]
    print(Cs_final)

    return Cs_final