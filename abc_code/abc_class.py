import logging
from time import time
import os
import numpy as np

import abc_code.global_var as g
from abc_code import utils
from abc_code import distance as dist
from params import output


class ABC(object):

    def __init__(self, abc, algorithm, N_params, N_proc, C_limits):

        self.N_params = N_params
        self.N_proc = N_proc
        self.N_each = algorithm['N_each']
        self.N_total = algorithm['N_total']
        self.N_params_in_task = algorithm['N_params_in_task']
        self.M = abc['num_training_points']
        self.C_limits = C_limits
        self.algorithm = algorithm

        if abc['algorithm'] == 'MCMC':  # MCMC
            logging.info('ABC algorithm: MCMC')
            self.C_array = utils.sampling_initial_for_MCMC(N_proc, C_limits, algorithm['eps'])
            self.main_loop = self.main_loop_MCMC
            self.work_func = work_function_MCMC
        elif abc['algorithm'] == 'IMCMC':  # IMCMC
            logging.info('ABC algorithm: IMCMC')
            self.main_loop = self.main_loop_IMCMC
            self.work_func = work_function_MCMC
            if self.N_each > 0:
                self.C_array = self.sampling(algorithm['sampling'], algorithm, C_limits)
            self.calibration = calibration_function_single_value
            if algorithm['N_params_in_task'] > 0:
                self.calibration = calibration_function_multiple_values
        elif abc['algorithm'] == 'AGM_MH':  # Gaussian mixture
            logging.info('ABC algorithm: Gaussian Mixture')
            self.C_array = utils.sampling_initial_for_gaussian_mixture(N_proc, algorithm['N_gaussians'],
                                                                       C_limits, algorithm['eps'])
            self.main_loop = self.main_loop_gaussian_mixture
            self.work_func = work_function_gaussian_mixture
        else:                   # Uniform
            logging.info('ABC algorithm: Uniform grid sampling')
            self.main_loop = self.main_loop_uniform
            self.C_array = self.sampling(algorithm['sampling'], algorithm, C_limits)
            self.work_func = work_function_single_value
            if algorithm['N_params_in_task'] != 0:
                self.work_func = work_function_multiple_values

    def sampling(self, sampling, abc_params, C_limits):
        if sampling == 'random':
            array = utils.sampling_random(abc_params['N_total'], C_limits)
        elif sampling == 'uniform':
            array = utils.sampling_uniform_grid(abc_params['N_each'],
                                                abc_params['N_params_in_task'], C_limits)
        elif sampling == 'sobol':
            array = utils.sampling_sobol(abc_params['N_total'], C_limits)
        logging.info('Sampling is {}'.format(sampling))
        return array

    def main_loop_IMCMC(self):

        if self.N_each > 0:    # Run calibration step
            logging.info('Calibration step with {} samples'.format(len(self.C_array)))

            start_calibration = time()
            S_init = []
            if g.par_process:
                g.par_process.run(func=self.calibration, tasks=self.C_array)
                S_init = g.par_process.get_results()
            else:
                from tqdm import tqdm
                with tqdm(total=self.N_total) as pbar:
                    for C in self.C_array:
                        S_init.append(self.calibration(C))
                        pbar.update()
                pbar.close()
            end_calibration = time()
            utils.timer(start_calibration, end_calibration, 'Time of calibration step')

            if self.N_params_in_task > 0:
                S_init = [chunk[:] for item in S_init for chunk in item]

            np.savez(os.path.join(output['output_path'],'calibration_all.npz'),  S_init=np.array(S_init))
            logging.info('Accepted parameters and distances saved in ./ABC/plots/calibration_all.npz')

        else:
            S_init = list(np.load('./plots/calibration_all.npz')['S_init'])

        logging.info('x = {}'.format(self.algorithm['x']))
        S_init.sort(key=lambda y: y[-1])
        S_init = np.array(S_init)
        g.eps = np.percentile(S_init, q=int(self.algorithm['x'] * 100), axis=0)[-1]
        logging.info('eps after calibration step = {}'.format(g.eps))

        S_init = S_init[np.where(S_init[:, -1] < g.eps)]
        g.std = self.algorithm['phi']*np.std(S_init[:, :-1], axis=0)
        logging.info('std for each parameter after calibration step:\n{}'.format(g.std))

        # Define new range
        for i in range(self.N_params):
            max_S = np.max(S_init[:, i])
            min_S = np.min(S_init[:, i])
            half_length = self.algorithm['phi'] * (max_S - min_S) / 2.0
            middle = (max_S + min_S) / 2.0
            self.C_limits[i] = np.array([middle - half_length, middle + half_length])
        logging.info('New parameters range after calibration step:\n{}'.format(self.C_limits))

        # Randomly choose starting points for Markov chains
        C_start = (S_init[np.random.choice(S_init.shape[0], self.N_proc, replace=False), :-1])
        np.set_printoptions(precision=3)
        logging.info('starting parameters for MCMC chains:\n{}'.format(C_start))
        self.C_array = C_start.tolist()
        np.savez('./plots/calibration.npz', C=S_init[:, :-1], dist=S_init[:, -1])
        logging.info('Accepted parameters and distances saved in ./ABC/plots/calibration.npz')
        exit()
        ####################################################################################################################
        # Markov chains
        self.main_loop_MCMC()

    def main_loop_MCMC(self):
        start = time()
        if g.par_process:
            g.par_process.run(func=self.work_func, tasks=self.C_array)
            result = g.par_process.get_results()
            end = time()
            g.accepted = np.array([chunk[:self.N_params] for item in result for chunk in item])
            g.dist = np.array([chunk[-1] for item in result for chunk in item])
        else:
            result = self.work_func(self.C_array[0])
            end = time()
            g.accepted = np.array([C[:self.N_params] for C in result if C])
            g.dist = np.array([C[-1] for C in result if C])
        utils.timer(start, end, 'Time ')
        logging.debug('Number of accepted parameters: {}'.format(len(g.accepted)))

    def main_loop_gaussian_mixture(self):
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
            with tqdm(total=self.N_each ** (self.N_params - self.N_params_in_task)) as pbar:
                for C in self.C_array:
                    result.append(self.work_func(C))
                    pbar.update()
            pbar.close()
        end = time()
        utils.timer(start, end, 'Time ')
        if self.N_params_in_task == 0:
            g.accepted = np.array([C[:self.N_params] for C in result if C])
            g.dist = np.array([C[-1] for C in result if C])
        else:
            g.accepted = np.array([chunk[:self.N_params] for item in result for chunk in item])
            g.dist = np.array([chunk[-1] for item in result for chunk in item])
        logging.info('Number of accepted values: {} {}%'.format(len(g.accepted),
                                                                round(len(g.accepted) / self.N_total * 100, 2)))


########################################################################################################################
# Work_functions
########################################################################################################################
dist_func = dist.distance_production_L2log


def calibration_function_single_value(C):
    """ Calibration function for IMCMC algorithm
        Accept all sampled values.
    :param C: list of sampled parameters
    :return:  list[Cs, dist]
    """
    distance = dist.calc_dist(C, dist_func)
    result = C[:]
    result.append(distance)
    return result


def calibration_function_multiple_values(C):
    return g.TEST_Model.sigma_from_C_calibration2(C, dist_func)


def work_function_single_value(C):
    """ Worker function for parallel regime (for pool.map from multiprocessing module)
    :param C: list of sampled parameters
    :return:  list[bool, Cs, dist], where bool=True, if values are accepted
    """
    distance = dist.calc_dist(C, dist_func)
    if distance <= g.eps:
        result = C[:]
        result.append(distance)
        return result


def work_function_multiple_values(C):
    """ Worker function for parallel regime (for pool.map from multiprocessing module)
    :param C: list of sampled parameters
    :return:  list[bool, Cs, dist], where bool=True, if values are accepted
    """
    return g.TEST_Model.sigma_from_C(C, dist_func)


def work_function_MCMC(C_init):

    N = g.N.chain
    C_limits = g.C_limits

    std = g.std
    result = []

    from tqdm import tqdm
    with tqdm(total=N) as pbar:

        # add first param
        distance = dist.calc_dist(C_init, dist_func)
        a = C_init[:]
        a.append(distance)
        result.append(a)
        pbar.update()
    ####################################################################################################################
        # Markov Chain
        counter_sample = 0
        counter_dist = 0
        for i in range(1, N):
            while True:
                while True:
                    # print(i, counter_dist, counter_sample)
                    if i < 50:
                        c = np.random.normal(result[-1][:-1], std)
                    else:
                        covariance_matrix = np.cov(np.array(result)[-50:, :-1].T)
                        c = np.random.multivariate_normal(result[-1][:-1], cov=covariance_matrix)
                    counter_sample += 1
                    if not(False in (C_limits[:, 0] < c) * (c < C_limits[:, 1])):
                        break
                distance = dist.calc_dist(c, dist_func)
                counter_dist += 1
                if distance <= g.eps:
                    a = list(c[:])
                    a.append(distance)
                    result.append(a)
                    pbar.update()
                    break
        pbar.close()
    print('Number of model and distance evaluations: {} ({} accepted)'.format(counter_dist, N))
    print('Number of sampling: {} ({} accepted)'.format(counter_sample, N))
    logging.info('Number of model and distance evaluations: {} ({} accepted)'.format(counter_dist, N))
    logging.info('Number of sampling: {} ({} accepted)'.format(counter_sample, N))
    return result


def work_function_gaussian_mixture(C_init):
    epsilon = 1e-6
    C_limits = g.C_limits
    result = []
    ############################################################################
    # Initialization
    ############################################################################
    # a)
    T_tot = g.N.chain
    # T_tot = 10
    T_train = int(100*g.N.params)
    T_stop = int(0.9*T_tot)
    assert T_train < T_stop/2, "T_train = {} is bigger then T_stop/2 = {}".format(T_train, T_stop/2)
    N = g.N.gaussians  # Number of Gaussians
    print(T_train, T_stop, T_tot, N)

    # b) Proposal
    mu = np.empty((N, g.N.params))
    cov = np.empty((N, g.N.params, g.N.params))
    for i in range(N):
        mu[i] = C_init[i]
    for i in range(N):
        cov[i] = np.diag(g.std**2)
    weights = np.ones(N)/N

    # c) Auxiliary parameters
    m = np.ones(N)

    ############################################################################
    # MH steps
    ############################################################################
    from tqdm import tqdm
    with tqdm(total=T_tot) as pbar:
        counter_sample = 0
        counter_dist = 0
        counter_update = 0
        for step in range(T_tot):
            if T_train < step < T_stop:
                ###########################
                #  Update proposal
                ###########################
                counter_update += 1
                # Find the closest Gaussian
                j = np.argmin(np.linalg.norm(mu - c, axis=1))
                m[j] += 1
                # update mu and cov
                mu[j] = 1 / m[j] * c + (m[j] - 1) / m[j] * mu[j]
                cov[j] = (np.outer(c - mu[j], (c - mu[j]).T) / m[j] + epsilon * np.identity(g.N.params))/(m[j] - 1) + \
                         (m[j] - 2) / (m[j] - 1) * cov[j]
                # update weights
                for i in range(N):
                    weights[i] = m[i] / (N + counter_update)

                # for j in range(N):
                #     lambda_, v = np.linalg.eig(cov[j])
                #     lambda_ = np.sqrt(lambda_)
                #     ell = Ellipse(xy=(mu[j, 0], mu[j, 1]),
                #                   width=lambda_[0], height=lambda_[1],
                #                   angle=np.rad2deg(np.arccos(v[0, 0])))
                #     # ell.set_facecolor('none')
                #     ax.add_artist(ell)
                #
                #     ax.scatter(mu[j, 0], mu[j, 1])
                # ax.axis([-1, 1, -1, 1])
                # fig.savefig('./plots/gaussian_mixture' + str(i))

                # mu, cov, weights = update_proposal(mu, cov, weights, m, c, i)

            while True:
                while True:
                    # print(i, counter_dist, counter_sample)

                    # Sample from gaussian mixture proposal
                    ind = np.random.choice(np.arange(N), p=weights)
                    c = np.random.multivariate_normal(mu[ind], cov=cov[ind])
                    counter_sample += 1

                    if not(False in (C_limits[:, 0] < c) * (c < C_limits[:, 1])):
                        break

                dist = dist.calc_dist(c, dist_func)
                counter_dist += 1
                if dist <= g.eps:
                    # prior_new = utils.get_prior(c)
                    # prior_old = utils.get_prior(result[-1][:-1])
                    # if prior_new == 0:
                    #     h = 0
                    # elif prior_old == 0:
                    #     h = 1
                    # else:
                    #     h = min(1, np.divide(prior_new, prior_old))  # np.divide return 0 for division by 0
                    #
                    # if h > 0 and np.random.random() < h:
                    a = list(c[:])
                    a.append(dist)
                    result.append(a)
                    pbar.update()
                    break
        pbar.close()
    print('Number of model and distance evaluations: {} ({} accepted)'.format(counter_dist, T_tot))
    print('Number of sampling: {} ({} accepted)'.format(counter_sample, T_tot))
    logging.info('Number of model and distance evaluations: {} ({} accepted)'.format(counter_dist, T_tot))
    logging.info('Number of sampling: {} ({} accepted)'.format(counter_sample, T_tot))
    return result

# ############
# #
# ############
# def work_function_PMC():
#
#     N_params = 1
#     print('N_params', N_params)
#     C_limits = params.C_limits
#     N = 10
#     M = 100
#     delta = 0.1
#     prior = 1/(-2*C_limits[0][0]**2 - (-2*C_limits[0][1]**2))
#     result = []
#     imp_var = (-2*C_limits[0][0]**2 - (-2*C_limits[0][1]**2))/5
#
#     def gausspdf(x, mu=0, sigma=1):
#         if sigma == 0:
#             print('sigma is 0')
#             exit()
#         u = ((x - mu) / np.abs(sigma))**2
#         y = np.exp(- u / 2) / (np.sqrt(2 * np.pi) * np.abs(sigma))
#         return y
#
#     ####################################################################################################################
#     def calc_dist(C):
#         # Generate data D
#         tau = g.TEST_Model.Reynolds_stresses_from_C(C)
#         # Calculate distance rho(pdf(D), pdf(D'))
#         dist = 0
#         for key in g.TEST_Model.elements_in_tensor:
#             pdf = np.histogram(tau[key].flatten(), bins=g.bins, range=g.domain, normed=1)[0]
#             d = distance_between_pdf_KL(pdf_modeled=pdf, key=key, axis=0)
#             dist += d
#         return dist
#     ####################################################################################################################
#     ####################################################################################################################
#     # first iteration
#     K = M
#     S_init = []
#     for i in range(M):
#         C = []
#         for i in range(N_params):
#             C.append(rand.uniform(C_limits[i][0], C_limits[i][1]))
#         C[0] = -2 * C[0] ** 2
#         dist = calc_dist(C)
#         a = C[:]
#         a.append(dist)
#         S_init.append(a)
#
#     S_init.sort(key=lambda x: x[-1])
#     S_init = np.array(S_init)
#     eps = np.percentile(S_init, q=75, axis=0)[-1]
#     print('eps = ', eps)
#     S_init = S_init[np.where(S_init[:, -1] < eps)]
#     # Calculate initial covariance
#     Cov = 2*np.std(S_init[:, 0])*np.ones(N)
#     print(Cov)
#     # Calculate initial weights
#     W_prev = np.ones(N)/N
#
#     t = 1
#     ####################################################################################################################
#     for n in range(10):
#         K = 0
#         t += 1
#         S = []
#         eps = S_prev[int(0.9*N), -1]
#         counter = 0
#         while len(S) < N:
#             K += 1
#             index = np.random.choice(N, p=W_prev)
#             mean = S_prev[index, 0]
#             C_new = rand.gauss(mean, Cov[index])
#             dist = calc_dist(C)
#             if dist <= eps:
#                 a = [C_new, dist]
#                 counter += 1
#                 S.append(a)
#                 K = 0
#         # end
#
#         S = np.array(S)
#         # set new weights
#         W = np.empty_like(W_prev)
#         for j in range(N):
#             denom = 0
#             for i in range(N):
#                 denom += W_prev[i]*gausspdf(S[j, 0], mu=S_prev[i, 0], sigma=Cov[i])
#             W[i] = prior/denom
#         W = W/np.sum(W)
#
#         # new covariance
#         Cov = 2 * np.std(S[:, 0]) * W
#         print(Cov)
#         # set next step
#         result.append(S)
#         S_prev = S.copy()
#         W_prev = W.copy()
#     return result



