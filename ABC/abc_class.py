import itertools
import logging
import random as rand
from time import time

import global_var as g
import params
import numpy as np
import utils
import sweep_params

class ABC(object):

    def __init__(self, N, C_limits, eps, C_array=None):

        self.N = N
        # self.N_total = N.each ** N.params
        self.M = N.training
        self.C_limits = C_limits
        self.eps = eps

        if params.PMC:
            self.work_func = work_function_PMC
            self.main_loop = self.main_loop_PMC
        elif params.MCMC == 1:  # MCMC
            logging.info('ABC algorithm: MCMC')
            self.C_array = self.form_C_array_initial_for_MCMC()
            self.main_loop = self.main_loop_MCMC
            self.work_func = work_function_MCMC
        elif params.MCMC == 2:  # IMCMC
            logging.info('ABC algorithm: IMCMC')
            if self.N.each > 0:
                if params.sweep:
                    self.C_array = sweep_params.random_sweep(self.N)
                else:
                    self.C_array = self.form_C_array_manual()
            self.main_loop = self.main_loop_IMCMC
            self.work_func = work_function_MCMC
            self.calibration = calibration_function_single_value
            if self.N.params_in_task > 0:
                self.calibration = calibration_function_multiple_values

        else:                   # Uniform
            logging.info('ABC algorithm: Uniform grid sampling')
            self.C_array = self.form_C_array_manual()
            self.main_loop = self.main_loop_uniform
            if N.params == 1 or N.params_in_task == 0:
                self.work_func = work_function_single_value
            else:
                self.work_func = work_function_multiple_values

    def form_C_array_initial_for_MCMC(self):

        C_array = []

        while len(C_array) <= self.N.proc:
            C = np.random.uniform(g.C_limits[:, 0], g.C_limits[:, 1])
            C_start = work_function_single_value(list(C))
            if C_start:
                C_array.append(C_start[:-1])
                logging.info('C_start = {}'.format(C_start[:-1]))

        return C_array

    # def form_C_array_random(self):
    #     """Create list of lists of N parameters uniformly distributed on given interval
    #     :return: list of lists of sampled parameters
    #     """
    #     C_array = []
    #     for i in range(self.N.each ** self.N.params):
    #         C = []
    #         for j in range(self.N.params):
    #             c = rand.uniform(self.C_limits[j][0], self.C_limits[j][1])
    #             C.append(-2*c**2)
    #         C_array.append(C)
    #
    #     return C_array


    def form_C_array_manual(self):
        """ Create list of lists of N parameters manually(make grid) uniformly distributed on given interval
        :return: list of lists of sampled parameters
        """
        if self.N.params == 1:
            C_array = []
            C1 = np.linspace(self.C_limits[0][0], self.C_limits[0][1], self.N.each + 1)
            C1 = C1[:-1] + (C1[1] - C1[0]) / 2
            C1 = -2 * C1 ** 2
            for i in C1:
                C_array.append([i])
        else:
            C = np.ndarray((self.N.params-self.N.params_in_task, self.N.each))
            for i in range(self.N.params-self.N.params_in_task):
                C[i, :] = utils.uniform_grid(self.C_limits[i], self.N.each)
            C[0] = -2 * C[0] ** 2
            permutation = itertools.product(*C)
            C_array = list(map(list, permutation))
        logging.debug('Form C_array manually: {} samples\n'.format(len(C_array)))
        return C_array


    def main_loop_PMC(self):
        start = time()
        result = self.work_func()
        end = time()
        utils.timer(start, end, 'Time ')
        g.accepted = np.array([chunk[:self.N.params] for item in result for chunk in item])
        g.dist = np.array([chunk[-1] for item in result for chunk in item])
        g.accepted[:, 0] = np.sqrt(-g.accepted[:, 0] / 2)   # return back to standard Cs (-2*Cs^2)
        print(g.accepted[:, 0])
        logging.debug('Number of accepted parameters: {}'.format(len(g.accepted)))

    def main_loop_IMCMC(self):

        if g.N.each > 0:    # Run calibration step
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

            if params.sweep:
                np.savez('./plots/sweep_params.npz', C=np.array(S_init)[:, :-1], dist=np.array(S_init)[:, -1])
                logging.info('Accepted parameters and distances saved in ./ABC/plots/sweep_params.npz')
                exit()
            else:
                np.savez('./plots/calibration_all.npz',  S_init=np.array(S_init))
                logging.info('Accepted parameters and distances saved in ./ABC/plots/calibration_all.npz')

        else:
            S_init = list(np.load('./plots/calibration_all.npz')['S_init'])

        x = 0.1
        phi = 1     # for range update
        logging.info('x = {}'.format(x))
        S_init.sort(key=lambda y: y[-1])
        S_init = np.array(S_init)
        g.eps = np.percentile(S_init, q=int(x * 100), axis=0)[-1]
        logging.info('eps after calibration step = {}'.format(g.eps))

        S_init = S_init[np.where(S_init[:, -1] < g.eps)]
        # result = np.array(S_init[:int(x*n)])
        g.std = np.std(S_init[:, :-1], axis=0)
        logging.info('std for each parameter after calibration step:\n{}'.format(g.std))
        for i in range(g.N.params):
            g.C_limits[i] = phi*[np.min(S_init[:, i]), np.max(S_init[:, i])]
        logging.info('New parameters range after calibration step:\n{}'.format(g.C_limits))

        # Randomly choose starting points for Markov chains
        C_start = (S_init[np.random.choice(S_init.shape[0], self.N.proc, replace=False), :-1])
        np.set_printoptions(precision=3)
        logging.info('starting parameters for MCMC chains:\n{}'.format(C_start))
        self.C_array = C_start.tolist()

        np.savez('./plots/calibration.npz', C=S_init[:, :-1], dist=S_init[:, -1])
        logging.info('Accepted parameters and distances saved in ./ABC/plots/calibration.npz')
        ####################################################################################################################
        # Markov chains
        self.main_loop_MCMC()

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
        g.accepted[:, 0] = np.sqrt(-g.accepted[:, 0] / 2)   # return back to standard Cs (-2*Cs^2)
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
        g.accepted[:, 0] = np.sqrt(-g.accepted[:, 0] / 2)
        # np.savetxt('accepted_'+str(N_params_in_task)+'.out', g.accepted)
        # np.savetxt('dist_'+str(N_params_in_task)+'.out', g.dist)
        logging.info('Number of accepted values: {} {}%'.format(len(g.accepted),
                                                                round(len(g.accepted) / self.N_total * 100, 2)))


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
    log_fill = np.empty_like(pdf_modeled)
    log_fill.fill(g.TINY_log)
    log_modeled = np.log(pdf_modeled, out=log_fill, where=pdf_modeled > g.TINY)
    dist = np.sum(np.multiply(g.TEST_sp.tau_pdf_true[key], (g.TEST_sp.log_tau_pdf_true[key] - log_modeled)), axis=axis)
    # dist = np.sum(np.multiply(pdf_modeled, (log_modeled - g.TEST_sp.log_tau_pdf_true[key])), axis=axis)

    return dist


def distance_between_pdf_L1log(pdf_modeled, key, axis=1):
    """Calculate statistical distance between two pdf as
    :param pdf_modeled: array of modeled pdf
    :return:            scalar of calculated distance
    """
    log_fill = np.empty_like(pdf_modeled)
    log_fill.fill(g.TINY_log)
    log_modeled = np.log(pdf_modeled, out=log_fill, where=pdf_modeled > g.TINY)
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
    log_fill = np.empty_like(pdf_modeled)
    log_fill.fill(g.TINY_log)
    log_modeled = np.log(pdf_modeled, out=log_fill, where=pdf_modeled > g.TINY)
    dist = np.mean((log_modeled - g.TEST_sp.log_tau_pdf_true[key]) ** 2, axis=axis)
    return dist


def distance_between_pdf_L2log(pdf_modeled, key, axis=1):
    """ Calculate statistical distance between two pdf.
    :param pdf_modeled:
    :param key:
    :param axis:
    :return:
    """
    log_fill = np.empty_like(pdf_modeled)
    log_fill.fill(g.TINY_log)
    log_modeled = np.log(pdf_modeled, out=log_fill, where=pdf_modeled > g.TINY)
    dist = np.sqrt(np.sum((log_modeled - g.TEST_sp.log_tau_pdf_true[key]) ** 2, axis=axis))
    return dist


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
        d = distance_between_pdf_KL(pdf_modeled=pdf, key=key, axis=0)
        dist += d
    result = C[:]
    result.append(dist)
    return result


def calibration_function_multiple_values(C):
    return g.TEST_Model.Reynolds_stresses_from_C_calibration2(C, distance_between_pdf_KL)

def work_function_single_value(C):
    """ Worker function for parallel regime (for pool.map from multiprocessing module)
    :param C: list of sampled parameters
    :return:  list[bool, Cs, dist], where bool=True, if values are accepted
    """
    tau = g.TEST_Model.Reynolds_stresses_from_C(C)

    dist = 0
    for key in g.TEST_Model.elements_in_tensor:
        pdf = np.histogram(tau[key].flatten(), bins=g.bins, range=g.domain, normed=1)[0]
        d = distance_between_pdf_KL(pdf_modeled=pdf, key=key, axis=0)
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
    return g.TEST_Model.Reynolds_stresses_from_C(C, distance_between_pdf_KL)


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
            d = distance_between_pdf_KL(pdf_modeled=pdf, key=key, axis=0)
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
        pbar.update()
    ####################################################################################################################
        # Markov Chain
        counter = 0
        for i in range(1, N):
            while True:
                while True:
                    c = np.random.normal(result[-1][:-1], std)
                    # if c[0] < 0:
                    if not (False in np.less(C_limits[:, 0], c) or False in np.less(c, C_limits[:, 1])):
                        break
                # c = np.random.normal(result[-1][:-1], std)
                dist = calc_dist(c)
                counter += 1
                if dist <= g.eps:
                    a = list(c[:])
                    a.append(dist)
                    result.append(a)
                    pbar.update()
                    break
        pbar.close()
    logging.info('Number of model and distance evaluations: {} ({} accepted)'.format(counter, N))
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
        Cov = 2 * np.std(S[:,0]) * W
        print(Cov)
        # set next step
        result.append(S)
        S_prev = S.copy()
        W_prev = W.copy()
    return result