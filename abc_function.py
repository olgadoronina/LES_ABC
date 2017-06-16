from params import *
import glob
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


def work_function(Cs):
    """ Worker function for parallel regime (for pool.map from multiprocessing module)
    :param Cs: scalar value of sampled parameter
    :return:   list[Cs, dist] if accepted and None if not
    """
    tau = glob.TEST.Reynolds_stresses_from_Cs(Cs)
    dist = 0
    for key, value in tau.items():
        x, y = utils.pdf_from_array(value.flatten(), bins, domain)
        dist += distance_between_pdf(pdf_modeled=y, pdf_true=glob.TEST.tau_pdf_true[key])
    if dist <= eps:
        return [Cs, dist]

def ABC():
    """ Approximate Beyasian Computation algorithm
    :return: scalar of best parameter value Cs
    """
    if not glob.TEST.tau_true:
        glob.TEST.Reynolds_stresses_from_DNS()
    for key, value in glob.TEST.tau_true.items():
        glob.TEST.tau_pdf_true[key] = utils.pdf_from_array(value, bins, domain)[1]

    if not glob.TEST.S_mod_S_ij:
        glob.TEST.strain_mod_strain_ij()

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
    plot.tau_abc(glob.TEST.tau_pdf_true, np.sort(result, axis=0)[:, 0], glob.TEST.field, glob.TEST.S_mod_S_ij)
    Cs_final = result[np.argmin(result[:, 1]), 0]
    return Cs_final