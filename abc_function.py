from params import *
import global_var as g
import utils
import plot
import parallel
import multiprocessing as mp

from time import sleep


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
    log_modeled = np.log(pdf_modeled, out=np.empty_like(pdf_modeled).fill(-20), where=pdf_modeled != 0)
    dist = np.sum(np.multiply(pdf_modeled, (log_modeled - log_true)))
    return dist


def work_function(C):
    """ Worker function for parallel regime (for pool.map from multiprocessing module)
    :param Cs: scalar value of sampled parameter
    :return:   list[Cs, dist] if accepted and None if not
    """
    tau = g.TEST_sp.Reynolds_stresses_from_C(C)
    dist = 0
    for key, value in tau.items():
        x, y = utils.pdf_from_array(value.flatten(), bins, domain)
        dist += distance_between_pdf(pdf_modeled=y, pdf_true=g.TEST_sp.tau_pdf_true[key])

    if dist <= g.eps:
        return [True, C, dist]
    else:
        return [False, C, dist]


def ABC(eps, N):
    """Approximate Beyasian Computation algorithm
    :return: scalar of best parameter value Cs
    """
    if not g.TEST_sp.tau_true:
        g.TEST_sp.Reynolds_stresses_from_DNS()
    for key, value in g.TEST_sp.tau_true.items():
        g.TEST_sp.tau_pdf_true[key] = utils.pdf_from_array(value, bins, domain)[1]
    if not g.TEST_sp.S:
        g.TEST_sp.S = utils.sparse_dict(g.TEST.S, g.TEST_sp.M[0])
    if not g.TEST_sp.Tensor_1:
        g.TEST_sp.calc_tensor_1()
    if ORDER > 1:
        if not g.TEST_sp.R:
            g.TEST_sp.R = utils.sparse_dict(g.TEST.R, g.TEST_sp.M[0])
        if not g.TEST_sp.Tensor_2:
            g.TEST_sp.calc_tensor_2()
        if not g.TEST_sp.Tensor_3:
            g.TEST_sp.calc_tensor_3()
        if not g.TEST_sp.Tensor_4:
            g.TEST_sp.calc_tensor_4()

    C_array = utils.form_C_array(g.TEST_sp.num_of_params)
    # Main loop
    ####################################################################################################################
    start = time()
    if PARALLEL:
        tasks = C_array
        par_process = parallel.Parallel(processes=N_proc)
        par_process.run(func=work_function, tasks=tasks)
        result = par_process.get_results()
    else:
        result = []
        for C in C_array:
            result.append(work_function((C, eps)))
    end = time()
    utils.timer(start, end, 'Time ')
    ####################################################################################################################
    # Marginal pdf
    C1_accepted = np.array([[C[0], dist] for [accepted, C, dist] in result if accepted])
    if ORDER > 1:
        C2_accepted = np.array([[C[1], dist] for [accepted, C, dist] in result if accepted])
        C3_accepted = np.array([[C[2], dist] for [accepted, C, dist] in result if accepted])
        C4_accepted = np.array([[C[3], dist] for [accepted, C, dist] in result if accepted])
    # if PLOT_ALL_DIST:
    #     C1_failed = np.array([[C[0], dist] for [accepted, C, dist] in result if not accepted])
    #     if ORDER > 1:
    #         C2_failed = np.array([[C[1], dist] for [accepted, C, dist] in result if not accepted])
    #         C3_failed = np.array([[C[2], dist] for [accepted, C, dist] in result if not accepted])
    #         C4_failed = np.array([[C[3], dist] for [accepted, C, dist] in result if not accepted])
    #     plot.Cs_scatter(C1_accepted, C1_failed, label=r'$C_s$')
    #     if ORDER > 1:
    #         plot.Cs_scatter(C2_accepted, C2_failed, label=r'$C_2$')
    #         plot.Cs_scatter(C3_accepted, C3_failed, label=r'$C_3$')
    #         plot.Cs_scatter(C4_accepted, C4_failed, label=r'$C_4$')

    # plot.Cs_scatter(C1_accepted, label=r'$C_s$')
    # if ORDER > 1:
    #     plot.Cs_scatter(C2_accepted, label=r'$C_2$')
    #     plot.Cs_scatter(C3_accepted, label=r'$C_3$')
    #     plot.Cs_scatter(C4_accepted, label=r'$C_4$')

    plot.histogram(C1_accepted[:, 0], bins=20, label=r'$C_s$')
    C_final = [C1_accepted[np.argmin(C1_accepted[:, 1]), 0]]
    ####################################################################################################################
    if ORDER > 1:
        plot.histogram(C2_accepted[:, 0], bins=20, label=r'$C_2$')
        plot.histogram(C3_accepted[:, 0], bins=20, label=r'$C_3$')
        plot.histogram(C4_accepted[:, 0], bins=20, label=r'$C_4$')
        minim = np.argmin(C1_accepted[:, 1])
        logging.debug('Minimum distance is in: ' + str(C1_accepted[minim, 0]) + ' '
                      + str(C2_accepted[minim, 0]) + ' ' + str(C3_accepted[minim, 0]) + ' '
                      + str(C4_accepted[minim, 0]))
        # Joint PDF
        C_joint = np.array([C for [accepted, C, dist] in result if accepted])
        logging.debug('Number of accepted values: ' + str(C_joint.shape[0]))
        H, edges = np.histogramdd(C_joint, bins=(10, 10, 10, 10))
        logging.debug('Max number in bin: ' + str(np.max(H)))
        logging.debug('Mean number in bin: ' + str(np.mean(H)))
        C1_bin = (edges[0][:-1] + edges[0][1:]) / 2
        if ORDER > 1:
            C2_bin = (edges[1][:-1] + edges[1][1:]) / 2
            C3_bin = (edges[2][:-1] + edges[2][1:]) / 2
            C4_bin = (edges[3][:-1] + edges[3][1:]) / 2
        i, j, k, m = np.unravel_index(H.argmax(), H.shape)
        logging.info('Estimated parameters from joint pdf: ' + str(C1_bin[i]) + ' ' + str(C2_bin[j]) + ' ' + str(
            C3_bin[k]) + ' ' + str(C4_bin[m]))
        C_final = [C1_bin[i], C2_bin[i], C3_bin[i], C4_bin[i]]
    ####################################################################################################################
    return C_final
