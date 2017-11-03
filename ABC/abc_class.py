import itertools
from time import time

import ABC.data as data
import ABC.global_var as g
import ABC.model as model
import ABC.parallel as parallel
import ABC.utils as utils
from ABC.params import *


class ABC(object):

    def __init__(self, N, M, eps, order):
        self.N = N
        self.M = M
        g.eps = eps
        self.order = order
        g.TEST_sp = data.DataSparse(g.TEST, M)
        g.TEST_Model = model.NonlinearModel(g.TEST_sp, self.order)
        self.num_of_params = g.TEST_Model.num_of_params
        self.C_array = self.form_C_array_manual(N_params-N_params_in_task)

        self.accepted = []
        self.dist = []
        self.C_final_dist = []
        self.C_final_joint = []
        if ORDER == 1 or N_params_in_task == 0:
            self.work_func = work_function_single_value
        elif N_params_in_task == 1 or N_params_in_task == 2:
            self.work_func = work_function_multiple_values
        logging.info('Number of samples per interval = {}'.format(N_each))
        logging.info('Number of parameters per task = {}'.format(N_params_in_task))


    def form_C_array_random(self):
        """Create list of lists of N parameters uniformly distributed on given interval
        :return: list of lists of sampled parameters
        """
        C_array = []
        for i in range(self.N):
            C = []
            for j in range(self.num_of_params):
                C.append(rand.uniform(C_limits[j][0], C_limits[j][1]))
            C_array.append(C)
        return C_array

    def form_C_array_manual(self, n=N_params):
        """ Create list of lists of N parameters manually(make grid) uniformly distributed on given interval
        :return: list of lists of sampled parameters
        """
        if n == 1:
            C_array = []
            C1 = np.linspace(C_limits[0][0], C_limits[0][1], N_each + 1)
            C1 = C1[:-1] + (C1[1] - C1[0]) / 2
            C1 = -2 * C1 ** 2
            for i in C1:
                C_array.append([i])
        else:
            if self.N != N_each**N_params:
                print('Achtung!: cannot manually sample C')
            C = np.ndarray((n, N_each))
            for i in range(n):
                C[i, :] = utils.uniform_grid(i)
            C[0] = -2*C[0]**2
            permutation = itertools.product(*C)
            C_array = list(map(list, permutation))
        return C_array

    def main_loop(self):
        """ Main loop of ABC algorithm, fill list of accepted parameters
        and list of distances (of accepted parameters)"""
        start = time()
        result = []
        if N_proc > 1:
            par_process = parallel.Parallel(processes=N_proc)
            par_process.run(func=self.work_func, tasks=self.C_array)
            result = par_process.get_results()
        else:
            with tqdm(total=N_each**(N_params-N_params_in_task)) as pbar:
                for C in self.C_array:
                    result.append(self.work_func(C))
                    pbar.update()
            pbar.close()
        end = time()
        utils.timer(start, end, 'Time ')
        if N_params_in_task == 0:
            self.accepted = np.array([C[:N_params] for C in result if C])
            self.dist = np.array([C[-1] for C in result if C])
        else:
            self.accepted = np.array([chunk[:N_params] for item in result for chunk in item])
            self.dist = np.array([chunk[-1] for item in result for chunk in item])
        self.accepted[:, 0] = np.sqrt(-self.accepted[:, 0] / 2)  # return back to standard Cs (-2*Cs^2)
        # np.savetxt('accepted_'+str(N_params_in_task)+'.out', self.accepted)
        # np.savetxt('dist_'+str(N_params_in_task)+'.out', self.dist)
        logging.info('Number of accepted values: {} {}%'.format(len(self.accepted),
                                                                round(len(self.accepted)/self.N*100, 2)))

    def calc_final_C(self):
        """ Estimate the best fit of parameters.
        For 1 parameter: based on minimum distance;
        For more then 1 parameter: based on joint pdf.
        """
        if len(self.accepted) == 0:
            logging.warning('No accepted values')
        else:
            if self.order == 1:
                self.C_final_joint = [self.accepted[:, 0][np.argmin(self.dist)]]
                logging.info('Estimated parameter:{}'.format(self.C_final_joint))
            else:
                H, edges = np.histogramdd(self.accepted, bins=num_bin_joint)
                logging.debug('Max number in bin: ' + str(np.max(H)))
                logging.debug('Mean number in bin: ' + str(np.mean(H)))
                edges = np.array(edges)
                C_bin = (edges[:, :-1] + edges[:, 1:]) / 2  # shift value in the center of the bin
                ind = np.argwhere(H == np.max(H))
                for i in ind:
                    point = []
                    for j in range(self.num_of_params):
                        point.append(C_bin[j, i[j]])
                    self.C_final_joint.append(point)
                if len(ind) > 10:
                    logging.warning('Can not estimate parameters from joint pdf!'
                                    'Too many bins ({} bins, max value {}) '
                                    'with the same max value in joint pdf'.format(len(ind), np.max(H)))
                else:
                    logging.info('Estimated parameters from joint pdf: {}'.format(self.C_final_joint))

    def plot_marginal_pdf(self):
        if len(self.accepted) == 0:
            logging.warning('No accepted values')
        else:
            minim = np.argmin(self.dist)
            self.C_final_dist = self.accepted[minim, :]
            logging.info('Minimum distance is {} in: {}'.format(self.dist[minim], self.C_final_dist))

            # Uncomment to make figure for each marginal pdf
            # for i in range(self.num_of_params):
            #     plt.hist(self.accepted[:, i], bins=N_each, normed=1, range=C_limits[i])
            #     plt.xlabel(params_names[i])
            #     plt.show()

            cmap = plt.cm.jet  # define the colormap
            cmaplist = [cmap(i) for i in range(cmap.N)]  # extract all colors from the .jet map
            cmaplist[0] = ('white')  # force the first color entry to be grey
            cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

            fig = plt.figure(figsize=(10, 10))
            for i in range(self.num_of_params):
                for j in range(self.num_of_params):
                    if i == j:
                        x, y = utils.pdf_from_array_with_x(self.accepted[:, i], bins=N_each, range=C_limits[i])
                        ax = plt.subplot2grid((self.num_of_params, self.num_of_params), (i, i))
                        ax.hist(self.accepted[:, i], bins=num_bin_joint, normed=1, alpha=0.5, color='grey',
                                 range=C_limits[i])
                        ax.plot(x, y)
                        ax.axis(xmin=C_limits[i, 0], xmax=C_limits[i, 1])
                        ax.set_xlabel(params_names[i])
                    elif i < j:
                        ax = plt.subplot2grid((self.num_of_params, self.num_of_params), (i, j))
                        ax.axis(xmin=C_limits[j, 0], xmax=C_limits[j, 1], ymin=C_limits[i, 0], ymax=C_limits[i, 1])
                        plt.hist2d(self.accepted[:, j], self.accepted[:, i], bins=num_bin_joint, cmap=cmap,
                                   range=[C_limits[j], C_limits[i]])

            fig.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.5)
            if FIGSHOW:
                plt.show()
            fig.savefig(plot_folder + 'marginal.pdf')
            del fig

    def plot_scatter(self):
        if len(self.accepted) == 0:
            logging.warning('No accepted values')
        else:
            for i in range(self.num_of_params):
                fig = plt.figure()
                ax = plt.axes()
                ax.axis(xmin=C_limits[i, 0], xmax=C_limits[i, 1], ymax=eps+1)
                ax.scatter(self.accepted[:, i], self.dist, color='blue')
                ax.set_xlabel(params_names[i])
                ax.set_ylabel(r'$\sum_{i,j}\rho(\widehat{T}_{ij}^{\mathcal{F}},\widehat{T}_{ij})$')
                fig.savefig(plot_folder + params_names[i] + '.pdf')
                if FIGSHOW:
                    plt.show()
                plt.close(fig)
            gc.collect()

    def plot_compare_tau(self, scale='LES'):

        fig, axarr = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(15, 6))
        if scale == 'LES':
            titles = [r'$\widetilde{\tau}_{11}$', r'$\widetilde{\tau}_{12}$', r'$\widetilde{\tau}_{13}$']
            if len(self.C_final_joint) == 1:
                tau_modeled_joint = model.NonlinearModel(g.LES, ORDER).Reynolds_stresses_from_C_tau(self.C_final_joint[0])
            tau_modeled_dist = model.NonlinearModel(g.LES, ORDER).Reynolds_stresses_from_C_tau(self.C_final_dist)
        if scale == 'TEST_M':
            titles = [r'$\widetilde{T}_{11}$', r'$\widetilde{T}_{12}$', r'$\widetilde{T}_{13}$']
            if len(self.C_final_joint) == 1:
                tau_modeled_joint = model.NonlinearModel(g.TEST_sp, ORDER).Reynolds_stresses_from_C_tau(self.C_final_joint[0])
            tau_modeled_dist = model.NonlinearModel(g.TEST_sp, ORDER).Reynolds_stresses_from_C_tau(self.C_final_dist)
        if scale == 'TEST':
            titles = [r'$\widetilde{T}_{11}$', r'$\widetilde{T}_{12}$', r'$\widetilde{T}_{13}$']
            if len(self.C_final_joint) == 1:
                tau_modeled_joint = model.NonlinearModel(g.TEST, ORDER).Reynolds_stresses_from_C_tau(self.C_final_joint[0])
            tau_modeled_dist = model.NonlinearModel(g.TEST, ORDER).Reynolds_stresses_from_C_tau(self.C_final_dist)
        for ind, key in enumerate(['uu', 'uv', 'uw']):
            if scale == 'LES':
                x, y = utils.pdf_from_array_with_x(g.LES.tau_true[key].flatten(), 100, domain)
            if scale == 'TEST':
                x, y = utils.pdf_from_array_with_x(g.TEST.tau_true[key].flatten(), 100, domain)
            if scale == 'TEST_M':
                x, y = utils.pdf_from_array_with_x(g.TEST_sp.tau_true[key].flatten(), 100, domain)
            axarr[ind].plot(x, y, 'r', linewidth=2, label='true')
            if len(self.C_final_joint) == 1:
                x, y = utils.pdf_from_array_with_x(tau_modeled_joint[key].flatten(), 100, domain)
                axarr[ind].plot(x, y, 'b', linewidth=2, label='modeled joint')
            x, y = utils.pdf_from_array_with_x(tau_modeled_dist[key].flatten(), 100, domain)
            axarr[ind].plot(x, y, 'g', linewidth=2, label='modeled dist')
            axarr[ind].set_xlabel(titles[ind])
        axarr[0].axis(xmin=domain[0], xmax=domain[1], ymin=1e-5)
        axarr[0].set_ylabel('pdf')
        axarr[0].set_yscale('log', nonposy='clip')
        fig.tight_layout()
        plt.legend(loc=0)
        if FIGSHOW:
            plt.show()
        fig.savefig(plot_folder + scale + '.pdf')
        del fig, axarr
        gc.collect()


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
    log_fill.fill(TINY_log)
    log_modeled = np.log(pdf_modeled, out=log_fill, where=pdf_modeled > TINY)
    # if np.isnan(np.sum(log_modeled)):
    #     print('log_modeled: nan is detected ')
    dist= np.sum(np.multiply(g.TEST_sp.tau_pdf_true[key], (g.TEST_sp.log_tau_pdf_true[key] - log_modeled)), axis=axis)
    return dist


def distance_between_pdf_L1log(pdf_modeled, key, axis=1):
    """Calculate statistical distance between two pdf as
    :param pdf_modeled: array of modeled pdf
    :return:            scalar of calculated distance
    """
    log_fill = np.empty_like(pdf_modeled)
    log_fill.fill(TINY_log)
    log_modeled = np.log(pdf_modeled, out=log_fill, where=pdf_modeled > TINY)
    dist = 0.5*np.sum(np.abs(log_modeled - g.TEST_sp.log_tau_pdf_true[key]), axis=axis)
    return dist


def distance_between_pdf_LSE(pdf_modeled, key, axis=1):
    """ Calculate statistical distance between two pdf as mean((P1-P2)^2).
    :param pdf_modeled: array of modeled pdf
    :param key: tensor component(key of dict)
    :param axis: equal 1 when pdf_modeled is 2D array
    :return: scalar or 1D array of calculated distance
    """
    dist = np.mean((pdf_modeled - g.TEST_sp.tau_pdf_true[key])**2, axis=axis)
    return dist


def distance_between_pdf_L2(pdf_modeled, key, axis=1):
    """ Calculate statistical distance between two pdf as sqrt(sum((P1-P2)^2)).
    :param pdf_modeled: array of modeled pdf
    :param key: tensor component(key of dict)
    :param axis: equal 1 when pdf_modeled is 2D array
    :return: scalar or 1D array of calculated distance
    """
    dist = np.sqrt(np.sum((pdf_modeled - g.TEST_sp.tau_pdf_true[key])**2, axis=axis))
    return dist


def distance_between_pdf_LSElog(pdf_modeled, key, axis=1):
    """ Calculate statistical distance between two pdf as mean((ln(P1)-ln(P2))^2).
        Function for N_params_in_task > 0
    :param pdf_modeled: array of modeled pdf
    :param key: tensor component(key of dict)
    :return: 1D array of calculated distance
    """
    log_fill = np.empty_like(pdf_modeled)
    log_fill.fill(TINY_log)
    log_modeled = np.log(pdf_modeled, out=log_fill, where=pdf_modeled > TINY)
    # if np.isnan(np.sum(log_modeled)):
    #     print('log_modeled: nan is detected ')
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
    log_fill.fill(TINY_log)
    log_modeled = np.log(pdf_modeled, out=log_fill, where=pdf_modeled > TINY)
    dist = np.sqrt(np.sum((log_modeled - g.TEST_sp.log_tau_pdf_true[key])**2, axis=axis))
    return dist
########################################################################################################################
## Work_functions
########################################################################################################################
def work_function_single_value(C):
    """ Worker function for parallel regime (for pool.map from multiprocessing module)
    :param C: list of sampled parameters
    :return:  list[bool, Cs, dist], where bool=True, if values are accepted
    """
    tau = g.TEST_Model.Reynolds_stresses_from_C(C)

    dist = 0
    for key in g.TEST_Model.elements_in_tensor:
        pdf = np.histogram(tau[key].flatten(), bins=bins, range=domain, normed=1)[0]
        d = distance_between_pdf_LSElog(pdf_modeled=pdf, key=key, axis=0)
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
    return g.TEST_Model.Reynolds_stresses_from_C(C, distance_between_pdf_LSElog)


