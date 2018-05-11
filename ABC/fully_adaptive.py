import numpy as np
import utils
# import matplotlib as mpl
# mpl.use('pdf')
# from matplotlib.patches import Ellipse
# import matplotlib.pyplot as plt
import logging
import global_var as g


epsilon = 1e-6



def calc_dist(C):
    # Generate data D
    tau = g.TEST_Model.Reynolds_stresses_from_C(C)
    # Calculate distance rho(pdf(D), pdf(D'))
    dist = 0
    for key in g.TEST_Model.elements_in_tensor:
        pdf = np.histogram(tau[key].flatten(), bins=g.bins, range=g.domain, normed=1)[0]
        d = utils.distance_between_pdf_L2log(pdf_modeled=pdf, key=key, axis=0)
        dist += d
    return dist


def work_function_gaussian_mixture(C_init):

    C_limits = g.C_limits
    result = []
    ############################################################################
    # Initialization
    ############################################################################
    # a)
    T_tot = g.N.chain
    # T_tot = 10
    T_train = 0.01*T_tot
    T_stop = 0.7*T_tot
    N = g.N.gaussians  # Number of Gaussians

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
    # S = np.empty((1, N, N_params))
    # S[0] = mu

    ############################################################################
    # MH steps
    ############################################################################
    from tqdm import tqdm
    with tqdm(total=T_tot) as pbar:
        # fig = plt.figure(figsize=(5, 3))
        # ax = plt.gca()
        counter_sample = 0
        counter_dist = 0
        for step in range(T_tot):
            if T_train < i < T_stop:
                ###########################
                #  Update proposal
                ###########################
                # Find the closest Gaussian
                j = np.argmin(np.linalg.norm(mu - c, axis=1))
                m[j] += 1
                # update S
                mu[j] = 1 / m[j] * c + (m[j] - 1) / m[j] * mu[j]
                cov[j] = (np.outer(c - mu[j],(c - mu[j]).T) / m[j] + epsilon * np.identity(g.N.params))/(m[j] - 1) + \
                         (m[j] - 2) / (m[j] - 1) * cov[j]
                for i in range(N):
                    weights[i] = m[i] / (step + N)

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

                dist = calc_dist(c)
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
