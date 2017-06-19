from params import *
import global_var as g
import random as rand
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
    log_modeled = np.log(pdf_modeled, out=np.zeros_like(pdf_modeled), where=pdf_modeled != 0)
    dist = np.sum(np.multiply(pdf_modeled, (log_modeled - log_true)))
    return dist


def work_function(tasks):
    """ Worker function for parallel regime (for pool.map from multiprocessing module)
    :param Cs: scalar value of sampled parameter
    :return:   list[Cs, dist] if accepted and None if not
    """
    Cs, eps = tasks[0], tasks[1]
    tau = g.TEST.Reynolds_stresses_from_Cs(Cs)
    dist = 0
    for key, value in tau.items():
        x, y = utils.pdf_from_array(value.flatten(), bins, domain)
        dist += distance_between_pdf(pdf_modeled=y, pdf_true=g.TEST.tau_pdf_true[key])
    if dist <= eps:
        return [True, Cs, dist]
    elif PLOT_ALL_DIST:
        return [False, Cs, dist]


def ABC(eps, N):
    """ Approximate Beyasian Computation algorithm
    :return: scalar of best parameter value Cs
    """
    if not g.TEST.tau_true:
        g.TEST.Reynolds_stresses_from_DNS()
    for key, value in g.TEST.tau_true.items():
        g.TEST.tau_pdf_true[key] = utils.pdf_from_array(value, bins, domain)[1]

    if not g.TEST.S_mod_S_ij:
        g.TEST.strain_mod_strain_ij()

    Cs_array = [rand.uniform(Cs_limits[0], Cs_limits[1]) for i in range(N)]
    # Main loop
    ####################################################################################################################
    start = time()
    if PARALLEL:
        print("\n%d workers" % mp.cpu_count())
        pool = mp.Pool(mp.cpu_count())
        tasks = [(Cs, eps) for Cs in Cs_array]
        result = pool.map(work_function, tasks)  # in parallel
        pool.close()
        pool.terminate()
    else:
        result = []
        for Cs in Cs_array:
            result.append(work_function(Cs))
    end = time()
    utils.timer(start, end, 'Time ')
    ####################################################################################################################
    Cs_accepted = np.array([[Cs, dist] for [accepted, Cs, dist] in result if accepted])
    if PLOT_ALL_DIST:
        Cs_failed = np.array([[Cs, dist] for [accepted, Cs, dist] in result if not accepted])
        plot.Cs_scatter(Cs_accepted, Cs_failed)

    plot.Cs_scatter(Cs_accepted)
    print(len(Cs_accepted))
    ####################################################################################################################
    # plot.tau_abc(np.sort(Cs_accepted, axis=0)[:, 0])
    Cs_final = Cs_accepted[np.argmin(Cs_accepted[:, 1]), 0]
    return Cs_final
