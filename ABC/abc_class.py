from ABC.params import *
import ABC.global_var as g
import ABC.data as data
import ABC.utils as utils
# import ABC.plot as plot
import ABC.parallel as parallel
import ABC.model as model
from tqdm import tqdm


class ABC(object):

    def __init__(self, N, M, eps, order):
        self.N = N
        self.M = M
        g.eps = eps
        self.order = order
        g.TEST_sp = data.DataSparse(g.TEST, M)
        g.TEST_Model = model.NonlinearModel(g.TEST_sp, self.order)
        self.num_of_params = g.TEST_Model.num_of_params
        self.C_array = self.form_C_array(N)
        self.result = []
        self.accepted = []
        self.dist = []
        self.C_final_dist = []
        self.C_final_joint = []

    def form_C_array(self, N):
        """Create list of lists of N parameters uniformly distributed on given interval
        :param N: number of samples values
        :param n: number of parameters
        :return: list of lists of sampled parameters
        """
        C_array = []
        for i in range(N):
            C = []
            for j in range(self.num_of_params):
                C.append(rand.uniform(C_limits[j][0], C_limits[j][1]))
            C_array.append(C)
        return C_array

    def main_loop(self):
        """ Main loop of ABC algorithm, fill list of accepted parameters """
        start = time()
        if PARALLEL:
            par_process = parallel.Parallel(processes=N_proc)
            par_process.run(func=work_function, tasks=self.C_array)
            self.result = par_process.get_results()
        else:
            with tqdm(total=N) as pbar:
                for C in self.C_array:
                    self.result.append(work_function(C))
                    pbar.update()
            pbar.close()
        end = time()
        utils.timer(start, end, 'Time ')
        self.accepted = np.array([C for [accepted, C, dist] in self.result if accepted])
        self.dist = np.array([dist for [accepted, C, dist] in self.result if accepted])
        logging.debug('Number of accepted values: ' + str(len(self.accepted)) + ' ' + str(round(len(self.accepted)/self.N*100,2))+'%')

    def plot_marginal_pdf(self):
        if len(self.accepted) == 0:
            logging.warning('No accepted values')
        else:
            minim = np.argmin(self.dist)
            self.C_final_dist = self.accepted[minim, :]
            logging.debug('Minimum distance is {} in: {}'.format(self.dist[minim], self.C_final_dist))

            # for i in range(self.num_of_params):
            #     plt.hist(self.accepted[:, i], bins=20, normed=1)
            #     plt.xlabel(params_names[i])
            #     plt.show()

            fig = plt.figure(figsize=(9, 9))
            for i in range(self.num_of_params):
                for j in range(self.num_of_params):
                    if i == j:
                        x, y = utils.pdf_from_array_with_x(self.accepted[:, i], bins=50, range=C_limits[i])
                        ax = plt.subplot2grid((self.num_of_params, self.num_of_params), (i, i))
                        plt.hist(self.accepted[:, i], bins=20, normed=1, alpha=0.5, color='grey')
                        plt.plot(x, y)
                        ax.axis(xmin=C_limits[i,0], xmax=C_limits[i,1])
                        ax.set_xlabel(params_names[i])
                    elif i < j:
                        ax = plt.subplot2grid((self.num_of_params, self.num_of_params), (i, j))
                        ax.axis(xmin=C_limits[j, 0], xmax=C_limits[j, 1], ymin=C_limits[i, 0], ymax=C_limits[i, 1])
                        plt.hist2d(self.accepted[:, j], self.accepted[:, i], bins=100, normed=1, cmap=plt.cm.jet)
            fig.tight_layout()
            # plt.legend(loc=0)
            fig.show()
            # del fig


    def plot_scatter(self):
        if len(self.accepted) == 0:
            logging.warning('No accepted values')
        else:
            for i in range(self.num_of_params):
                plt.axis(xmin=C_limits[i, 0], xmax=C_limits[i, 1], ymax=eps+1)
                plt.scatter(self.accepted[:, i], self.dist, color='blue')
                plt.xlabel(params_names[i])
                plt.ylabel(r'$\sum_{i,j}\rho(\widehat{T}_{ij}^{\mathcal{F}},\widehat{T}_{ij})$')
                plt.show()
            gc.collect()

    def plot_compare_tau(self, scale='LES'):

        fig, axarr = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(15, 6))
        if scale == 'LES':
            titles = [r'$\widetilde{\tau}_{11}$', r'$\widetilde{\tau}_{12}$', r'$\widetilde{\tau}_{13}$']
            tau_modeled_joint = model.NonlinearModel(g.LES, ORDER).Reynolds_stresses_from_C(self.C_final_joint)
            tau_modeled_dist = model.NonlinearModel(g.LES, ORDER).Reynolds_stresses_from_C(self.C_final_dist)
        if scale == 'TEST':
            titles = [r'$\widetilde{T}_{11}$', r'$\widetilde{T}_{12}$', r'$\widetilde{T}_{13}$']
            tau_modeled_joint = model.NonlinearModel(g.TEST, ORDER).Reynolds_stresses_from_C(self.C_final_joint)
            tau_modeled_dist = model.NonlinearModel(g.TEST, ORDER).Reynolds_stresses_from_C(self.C_final_dist)
        for ind, key in enumerate(['uu', 'uv', 'uw']):
            x, y = utils.pdf_from_array_with_x(g.LES.tau_true[key].flatten(), 100, domain)
            axarr[ind].plot(x, y, 'r', linewidth=2, label='true')
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
        plt.show()
        del fig, axarr
        gc.collect()

    def calc_final_C(self):
        """ Estimate the best fit of parameters based on joint pdf if more than one parameter. """
        # Joint PDF
        if len(self.accepted) == 0:
            logging.warning('No accepted values')
        else:
            if self.order == 1:
                self.C_final_joint = [self.accepted[:, 0][np.argmin(self.dist)]]
                logging.info('Estimated parameters:{}'.format(self.C_final_joint))
            else:
                C_joint = self.accepted
                H, edges = np.histogramdd(C_joint, bins=num_bin_joint)
                logging.debug('Max number in bin: ' + str(np.max(H)))
                logging.debug('Mean number in bin: ' + str(np.mean(H)))
                edges = np.array(edges)
                C_bin = (edges[:, :-1] + edges[:, 1:]) / 2  # shift value in the center of the bin
                ind = np.unravel_index(H.argmax(), H.shape)
                for i in range(self.num_of_params):
                    self.C_final_joint.append(C_bin[i, ind[i]])
                logging.info('Estimated parameters from joint pdf: {}'.format(self.C_final_joint))


def distance_between_pdf(pdf_modeled, key):
    """Calculate statistical distance between two pdf as
    the Kullback-Leibler (KL) divergence (no symmetry).
    :param pdf_modeled: array of modeled pdf
    :return:            scalar of calculated distance
    """
    log_modeled = np.log(pdf_modeled, out=np.empty_like(pdf_modeled).fill(-20), where=pdf_modeled != 0)
    dist = np.sum(np.multiply(pdf_modeled, (log_modeled - g.TEST_sp.log_tau_pdf_true[key])))
    return dist


def work_function(C):
    """ Worker function for parallel regime (for pool.map from multiprocessing module)
    :param C: list of sampled parameters
    :return:  list[bool, Cs, dist], where bool=True, if values are accepted
    """
    tau = g.TEST_Model.Reynolds_stresses_from_C(C)
    dist = 0
    for key in g.TEST_Model.elements_in_tensor:
        pdf, edges = np.histogram(tau[key].flatten(), bins=bins, range=domain, normed=1)
        dist += distance_between_pdf(pdf_modeled=pdf, key=key)
    if dist <= g.eps:
        return [True, C, dist]
    else:
        return [False, C, dist]
