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


def work_function(Cs):
    """ Worker function for parallel regime (for pool.map from multiprocessing module)
    :param Cs: scalar value of sampled parameter
    :return:   list[Cs, dist] if accepted and None if not
    """
    tau = calculate.Reynolds_stresses_from_Cs(TEST, Cs, TEST_delta, S_mod_S_ij_tmp=S_mod_S_ij)
    dist = 0
    for key, value in tau.items():
        x, y = utils.pdf_from_array(value.flatten(), bins, domain)
        dist += distance_between_pdf(pdf_modeled=y, pdf_true=pdf_true[key])

    if dist <= eps:
        return [Cs, dist]


def ABC(tau_exact, test_field):
    """ Approximate Beyasian Computation algorithm
    :param tau_exact:  dictionary of true data
    :param test_field: dictionary of the data on TEST scale
    :return:           scalar of best parameter value Cs
    """
    global pdf_true
    global S_mod_S_ij


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
        result = pool.map(work_function, Cs_array)  # in parallel
        pool.terminate()
    else:
        result = []
        for Cs in Cs_array:
            result.append(work_function(Cs))
    end = time()
    utils.timer(start, end, 'Time ')
    result = np.array([x for x in result if x is not None])
    ####################################################################################################################
    plot.Cs_scatter(result[:, 0], result[:, 1])
    plot.tau_abc(pdf_true, np.sort(result, axis=0)[:, 0], test_field, S_mod_S_ij)
    Cs_final = result[np.argmin(result[:, 1]), 0]
    return Cs_final