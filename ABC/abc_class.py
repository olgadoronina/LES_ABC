import logging
import random as rand
from time import time

import global_var as g
import params
import numpy as np
import utils


class ABC(object):

    def __init__(self, N, form_C_array):

        self.N = N
        self.M = N.training

        self.C_array = form_C_array()

        if params.MCMC == 1:  # MCMC
            logging.info('ABC algorithm: MCMC')
            self.main_loop = self.main_loop_MCMC
            self.work_func = work_function_MCMC
        elif params.MCMC == 2:  # IMCMC
            logging.info('ABC algorithm: IMCMC')
            self.main_loop = self.main_loop_IMCMC
            self.work_func = work_function_MCMC
            self.calibration = calibration_function_single_value
            if self.N.params_in_task > 0:
                self.calibration = calibration_function_multiple_values
        else:                   # Uniform
            logging.info('ABC algorithm: Uniform grid sampling')
            self.main_loop = self.main_loop_uniform
            if N.params == 1 or N.params_in_task == 0:
                self.work_func = work_function_single_value
            else:
                self.work_func = work_function_multiple_values

    def main_loop_IMCMC(self):

        if self.N.each > 0:    # Run calibration step
            logging.info('Calibration step with {} samples'.format(len(self.C_array)))

            start_calibration = time()
            S_init = []
            if g.par_process:
                g.par_process.run(func=self.calibration, tasks=self.C_array)
                S_init = g.par_process.get_results()
            else:
                from tqdm import tqdm
                with tqdm(total=self.N.calibration) as pbar:
                    for C in self.C_array:
                        S_init.append(self.calibration(C))
                        pbar.update()
                pbar.close()
            end_calibration = time()
            utils.timer(start_calibration, end_calibration, 'Time of calibration step')

            if self.N.params_in_task > 0:
                S_init = [chunk[:] for item in S_init for chunk in item]

            np.savez('./plots/calibration_all.npz',  S_init=np.array(S_init))
            logging.info('Accepted parameters and distances saved in ./ABC/plots/calibration_all.npz')

        else:
            S_init = list(np.load('./plots/calibration_all.npz')['S_init'])

        x = 0.15
        phi = 1     # for range update
        logging.info('x = {}'.format(x))
        S_init.sort(key=lambda y: y[-1])
        S_init = np.array(S_init)
        g.eps = np.percentile(S_init, q=int(x * 100), axis=0)[-1]
        logging.info('eps after calibration step = {}'.format(g.eps))

        S_init = S_init[np.where(S_init[:, -1] < g.eps)]
        g.std = phi*np.std(S_init[:, :-1], axis=0)
        logging.info('std for each parameter after calibration step:\n{}'.format(g.std))
        for i in range(g.N.params):
            max_S = np.max(S_init[:, i])
            min_S = np.min(S_init[:, i])
            half_length = phi * (max_S - min_S) / 2.0
            middle = (max_S + min_S) / 2.0
            g.C_limits[i] = np.array( [middle - half_length, middle + half_length])
        logging.info('New parameters range after calibration step:\n{}'.format(g.C_limits))

        # Randomly choose starting points for Markov chains
        C_start = (S_init[np.random.choice(S_init.shape[0], self.N.proc, replace=False), :-1])
        np.set_printoptions(precision=3)
        logging.info('starting parameters for MCMC chains:\n{}'.format(C_start))
        self.C_array = C_start.tolist()
        # Save prior
        d = (g.C_limits[:, 1] - g.C_limits[:, 0]) / g.N.each
        limits = g.C_limits.copy()
        limits[:, 1] += d # to calculate prior on right edge
        print(tuple(map(tuple, limits)))
        print(S_init[:, :-1].shape)
        g.prior, _ = np.histogramdd(S_init[:, :-1], bins=g.N.each+1, normed=True, range=tuple(map(tuple, limits)))
        print(g.prior.shape)
        np.savez('./plots/prior.npz', prior=g.prior, C_limits=g.C_limits)
        # S_init[:, 0] = np.sqrt(-S_init[:, 0] / 2)   # return back to standard Cs (-2*Cs^2)
        np.savez('./plots/calibration.npz', C=S_init[:, :-1], dist=S_init[:, -1])
        logging.info('Accepted parameters and distances saved in ./ABC/plots/calibration.npz')
        ####################################################################################################################
        # Markov chains
        self.main_loop_MCMC()
        # g.accepted = np.vstack((g.accepted, S_init))

    def main_loop_MCMC(self):
        start = time()
        if g.par_process:
            g.par_process.run(func=self.work_func, tasks=self.C_array)
            result = g.par_process.get_results()
            end = time()
            g.accepted = np.array([chunk[:self.N.params] for item in result for chunk in item])
            g.dist = np.array([chunk[-1] for item in result for chunk in item])
        else:
            result = self.work_func(self.C_array[0])
            end = time()
            g.accepted = np.array([C[:self.N.params] for C in result if C])
            g.dist = np.array([C[-1] for C in result if C])
        utils.timer(start, end, 'Time ')
        # g.accepted[:, 0] = np.sqrt(-g.accepted[:, 0] / 2)   # return back to standard Cs (-2*Cs^2)
        logging.debug('Number of accepted parameters: {}'.format(len(g.accepted)))

    def main_loop_uniform(self):
        """ Main loop of ABC algorithm, fill list of accepted parameters
        and list of distances (of accepted parameters)"""
        start = time()
        result = []
        if g.par_process:
            g.par_process.run(func=self.work_func, tasks=self.C_array)
            result = g.par_process.get_results()
        else:
            from tqdm import tqdm
            with tqdm(total=self.N.each ** (self.N.params - self.N.params_in_task)) as pbar:
                for C in self.C_array:
                    result.append(self.work_func(C))
                    pbar.update()
            pbar.close()
        end = time()
        utils.timer(start, end, 'Time ')
        if self.N.params_in_task == 0:
            g.accepted = np.array([C[:self.N.params] for C in result if C])
            g.dist = np.array([C[-1] for C in result if C])
        else:
            g.accepted = np.array([chunk[:self.N.params] for item in result for chunk in item])
            g.dist = np.array([chunk[-1] for item in result for chunk in item])
        # g.accepted[:, 0] = np.sqrt(-g.accepted[:, 0] / 2)
        # np.savetxt('accepted_'+str(N_params_in_task)+'.out', g.accepted)
        # np.savetxt('dist_'+str(N_params_in_task)+'.out', g.dist)
        logging.info('Number of accepted values: {} {}%'.format(len(g.accepted),
                                                                round(len(g.accepted) / self.N.total * 100, 2)))


########################################################################################################################
# Work_functions
########################################################################################################################
def calibration_function_single_value(C):
    """ Calibration function for IMCMC algorithm
        Accept all sampled values.
    :param C: list of sampled parameters
    :return:  list[Cs, dist]
    """
    tau = g.TEST_Model.Reynolds_stresses_from_C(C)
    dist = 0
    for key in g.TEST_Model.elements_in_tensor:
        pdf = np.histogram(tau[key].flatten(), bins=g.bins, range=g.domain, normed=1)[0]
        d = distance_between_pdf_L2log(pdf_modeled=pdf, key=key, axis=0)
        dist += d
    result = C[:]
    result.append(dist)
    return result


def calibration_function_multiple_values(C):
    return g.TEST_Model.Reynolds_stresses_from_C_calibration2(C, distance_between_pdf_L2log)


def work_function_single_value(C):
    """ Worker function for parallel regime (for pool.map from multiprocessing module)
    :param C: list of sampled parameters
    :return:  list[bool, Cs, dist], where bool=True, if values are accepted
    """
    tau = g.TEST_Model.Reynolds_stresses_from_C(C)

    dist = 0
    for key in g.TEST_Model.elements_in_tensor:
        pdf = np.histogram(tau[key].flatten(), bins=g.bins, range=g.domain, normed=1)[0]
        d = distance_between_pdf_L2log(pdf_modeled=pdf, key=key, axis=0)
        dist += d
    if dist <= g.eps:
        result = C[:]
        result.append(dist)
        return result


def work_function_multiple_values(C):
    """ Worker function for parallel regime (for pool.map from multiprocessing module)
    :param C: list of sampled parameters
    :return:  list[bool, Cs, dist], where bool=True, if values are accepted
    """
    return g.TEST_Model.Reynolds_stresses_from_C(C, distance_between_pdf_L2log)


def work_function_MCMC(C_init):

    N = g.N.chain
    C_limits = g.C_limits

    std = g.std
    result = []

    ####################################################################################################################
    def calc_dist(C):
        # Generate data D
        tau = g.TEST_Model.Reynolds_stresses_from_C(C)
        # Calculate distance rho(pdf(D), pdf(D'))
        dist = 0
        for key in g.TEST_Model.elements_in_tensor:
            pdf = np.histogram(tau[key].flatten(), bins=g.bins, range=g.domain, normed=1)[0]
            d = distance_between_pdf_L2log(pdf_modeled=pdf, key=key, axis=0)
            dist += d
        return dist
    ####################################################################################################################

    from tqdm import tqdm
    with tqdm(total=N) as pbar:

        # add first param
        dist = calc_dist(C_init)
        a = C_init[:]
        a.append(dist)
        result.append(a)
        # print(result)
        # print(result[-1][:-1])
        pbar.update()
    ####################################################################################################################
        # Markov Chain
        counter_sample = 0
        counter_dist = 0
        for i in range(1, N):
            while True:
                while True:
                    # print(i, counter_dist, counter_sample)
                    if i < 10:
                        c = np.random.normal(result[-1][:-1], std)
                    else:
                        # covariance_matrix = np.cov(np.array(result)[-10:, :-1].T)
                        # c = np.random.multivariate_normal(result[-1][:-1], cov=covariance_matrix)
                        c = np.random.normal(result[-1][:-1], std)

                    counter_sample += 1
                    if not(False in (g.C_limits[:, 0] < c) * (c < g.C_limits[:, 1])):
                        break
                dist = calc_dist(c)
                counter_dist += 1
                if dist <= g.eps:
                    prior_new = utils.get_prior(c)
                    prior_old = utils.get_prior(result[-1][:-1])
                    if prior_new == 0:
                        h = 0
                    elif prior_old == 0:
                        h = 1
                    else:
                        h = min(1, np.divide(prior_new, prior_old))  # np.divide return 0 for division by 0

                    if h > 0 and np.random.random() < h:
                        a = list(c[:])
                        a.append(dist)
                        result.append(a)
                        pbar.update()
                        break
        pbar.close()
    print('Number of model and distance evaluations: {} ({} accepted)'.format(counter_dist, N))
    print('Number of sampling: {} ({} accepted)'.format(counter_sample, N))
    logging.info('Number of model and distance evaluations: {} ({} accepted)'.format(counter_dist, N))
    logging.info('Number of sampling: {} ({} accepted)'.format(counter_sample, N))
    return result


############
#
############
def work_function_PMC():

    N_params = 1
    print('N_params', N_params)
    C_limits = params.C_limits
    N = 10
    M = 100
    delta = 0.1
    prior = 1/(-2*C_limits[0][0]**2 - (-2*C_limits[0][1]**2))
    result = []
    imp_var = (-2*C_limits[0][0]**2 - (-2*C_limits[0][1]**2))/5

    def gausspdf(x, mu=0, sigma=1):
        if sigma == 0:
            print('sigma is 0')
            exit()
        u = ((x - mu) / np.abs(sigma))**2
        y = np.exp(- u / 2) / (np.sqrt(2 * np.pi) * np.abs(sigma))
        return y

    ####################################################################################################################
    def calc_dist(C):
        # Generate data D
        tau = g.TEST_Model.Reynolds_stresses_from_C(C)
        # Calculate distance rho(pdf(D), pdf(D'))
        dist = 0
        for key in g.TEST_Model.elements_in_tensor:
            pdf = np.histogram(tau[key].flatten(), bins=g.bins, range=g.domain, normed=1)[0]
            d = distance_between_pdf_KL(pdf_modeled=pdf, key=key, axis=0)
            dist += d
        return dist
    ####################################################################################################################
    ####################################################################################################################
    # first iteration
    K = M
    S_init = []
    for i in range(M):
        C = []
        for i in range(N_params):
            C.append(rand.uniform(C_limits[i][0], C_limits[i][1]))
        C[0] = -2 * C[0] ** 2
        dist = calc_dist(C)
        a = C[:]
        a.append(dist)
        S_init.append(a)

    S_init.sort(key=lambda x: x[-1])
    S_init = np.array(S_init)
    eps = np.percentile(S_init, q=75, axis=0)[-1]
    print('eps = ', eps)
    S_init = S_init[np.where(S_init[:, -1] < eps)]
    # Calculate initial covariance
    Cov = 2*np.std(S_init[:, 0])*np.ones(N)
    print(Cov)
    # Calculate initial weights
    W_prev = np.ones(N)/N

    t = 1
    ####################################################################################################################
    for n in range(10):
        K = 0
        t += 1
        S = []
        eps = S_prev[int(0.9*N), -1]
        counter = 0
        while len(S) < N:
            K += 1
            index = np.random.choice(N, p=W_prev)
            mean = S_prev[index, 0]
            C_new = rand.gauss(mean, Cov[index])
            dist = calc_dist(C)
            if dist <= eps:
                a = [C_new, dist]
                counter += 1
                S.append(a)
                K = 0
        # end

        S = np.array(S)
        # set new weights
        W = np.empty_like(W_prev)
        for j in range(N):
            denom = 0
            for i in range(N):
                denom += W_prev[i]*gausspdf(S[j, 0], mu=S_prev[i, 0], sigma=Cov[i])
            W[i] = prior/denom
        W = W/np.sum(W)

        # new covariance
        Cov = 2 * np.std(S[:, 0]) * W
        print(Cov)
        # set next step
        result.append(S)
        S_prev = S.copy()
        W_prev = W.copy()
    return result



########################################################################################################################
## Distance functions
########################################################################################################################
def distance_between_pdf_KL(pdf_modeled, key, axis=1):
    """Calculate statistical distance between two pdf as
    the Kullback-Leibler (KL) divergence (no symmetry).
    Function for N_params_in_task > 0
    :param pdf_modeled: array of modeled pdf
    :param key: tensor component(key of dict)
    :return: 1D array of calculated distance
    """

    log_modeled = utils.take_safe_log(pdf_modeled)
    dist = np.sum(np.multiply(g.TEST_sp.tau_pdf_true[key], (g.TEST_sp.log_tau_pdf_true[key] - log_modeled)), axis=axis)

    return dist


def distance_between_pdf_L1log(pdf_modeled, key, axis=1):
    """Calculate statistical distance between two pdf as
    :param pdf_modeled: array of modeled pdf
    :return:            scalar of calculated distance
    """

    log_modeled = utils.take_safe_log(pdf_modeled)
    dist = 0.5 * np.sum(np.abs(log_modeled - g.TEST_sp.log_tau_pdf_true[key]), axis=axis)
    return dist


def distance_between_pdf_LSE(pdf_modeled, key, axis=1):
    """ Calculate statistical distance between two pdf as mean((P1-P2)^2).
    :param pdf_modeled: array of modeled pdf
    :param key: tensor component(key of dict)
    :param axis: equal 1 when pdf_modeled is 2D array
    :return: scalar or 1D array of calculated distance
    """
    dist = np.mean((pdf_modeled - g.TEST_sp.tau_pdf_true[key]) ** 2, axis=axis)
    return dist


def distance_between_pdf_L2(pdf_modeled, key, axis=1):
    """ Calculate statistical distance between two pdf as sqrt(sum((P1-P2)^2)).
    :param pdf_modeled: array of modeled pdf
    :param key: tensor component(key of dict)
    :param axis: equal 1 when pdf_modeled is 2D array
    :return: scalar or 1D array of calculated distance
    """
    dist = np.sqrt(np.sum((pdf_modeled - g.TEST_sp.tau_pdf_true[key]) ** 2, axis=axis))
    return dist


def distance_between_pdf_LSElog(pdf_modeled, key, axis=1):
    """ Calculate statistical distance between two pdf as mean((ln(P1)-ln(P2))^2).
    :param pdf_modeled: array of modeled pdf
    :param key: tensor component(key of dict)
    :return: 1D array of calculated distance
    """
    log_modeled = utils.take_safe_log(pdf_modeled)
    dist = np.mean((log_modeled - g.TEST_sp.log_tau_pdf_true[key]) ** 2, axis=axis)
    return dist


def distance_between_pdf_L2log(pdf_modeled, key, axis=1):
    """ Calculate statistical distance between two pdf.
    :param pdf_modeled:
    :param key:
    :param axis:
    :return:
    """
    log_modeled = utils.take_safe_log(pdf_modeled)
    dist = np.sum((log_modeled - g.TEST_sp.log_tau_pdf_true[key]) ** 2, axis=axis)
    return dist

