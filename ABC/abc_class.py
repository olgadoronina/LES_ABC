from ABC.params import *
import ABC.global_var as g
import ABC.data as data
import ABC.utils as utils
# import ABC.plot as plot
import ABC.parallel as parallel
import ABC.model as model
from tqdm import tqdm
import itertools
import baseconvert


class ABC(object):

    def __init__(self, N, M, eps, order):
        self.N = N
        self.M = M
        g.eps = eps
        self.order = order
        g.TEST_sp = data.DataSparse(g.TEST, M)
        g.TEST_Model = model.NonlinearModel(g.TEST_sp, self.order)
        self.num_of_params = g.TEST_Model.num_of_params
        if ORDER == 1:
            self.C_array = self.form_Cs_array_manual()
        elif ORDER == 2:
            self.C_array = self.form_C_array_manual()
        self.result = []
        self.accepted = []
        self.dist = []
        self.C_final_dist = []
        self.C_final_joint = []

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

    def form_Cs_array_manual(self):
        """ Create list of lists of N parameters manually(make grid) uniformly distributed on given interval
            for one parametes Smagorinsky model.
        :return: list of lists of sampled parameters
        """
        C_array = []
        C1 = np.linspace(C_limits[0][0], C_limits[0][1], N_each + 1)
        C1 = C1[:-1] + (C1[1] - C1[0]) / 2
        for i in C1:
            C_array.append([i])
        return C_array

    def form_C_array_manual(self, iter=None):
        """ Create list of lists of N parameters manually(make grid) uniformly distributed on given interval
        :return: list of lists of sampled parameters
        """
        if self.N != N_each**N_params:
            print('Achtung!: cannot manually sample C')

        elif ORDER == 2:
            C = np.ndarray((N_params, N_each))
            for i in range(N_params):
                C_tmp = np.linspace(C_limits[i][0], C_limits[i][1], N_each+1)
                C[i, :] = C_tmp[:-1] + (C_tmp[:1] - C_tmp[0])/2
            permitation = itertools.product(*C)
            C_array = list(map(list, permitation))

        else:
            index = baseconvert(iter, 10, 40)
            print('index = ',index)
            C = np.ndarray((N_params, N_each))
            for i in range(N_params - N_params_in_pool):
                C_tmp = np.linspace(C_limits[i][0], C_limits[i][1], N_each + 1)
                C[i, :] = C_tmp[:-1] + (C_tmp[:1] - C_tmp[0]) / 2
            for i in range(N_params_in_pool, N_params):
                C_tmp = np.linspace(C_limits[i][0], C_limits[i][1], N_each + 1)

            permitation = itertools.product(*C)
            C_array = list(map(list, permitation))

        return C_array

    def main_loop(self):
        """ Main loop of ABC algorithm, fill list of accepted parameters """
        start = time()
        if PARALLEL:
            if ORDER <= 2:
                par_process = parallel.Parallel(processes=N_proc)
                par_process.run(func=work_function, tasks=self.C_array)
                self.result = par_process.get_results()
            else:
                for i in range(N_each**(N_params-N_params_in_pool)):
                    C_array = self.form_C_array_manual()
                    par_process = parallel.Parallel(processes=N_proc)
                    par_process.run(func=work_function, tasks=C_array)
                    self.result.append(par_process.get_results())
        else:
            with tqdm(total=self.N) as pbar:
                for C in self.C_array:
                    self.result.append(work_function(C))
                    pbar.update()
            pbar.close()
        end = time()
        utils.timer(start, end, 'Time ')
        self.accepted = np.array([C for [accepted, C, dist] in self.result if accepted])
        self.dist = np.array([dist for [accepted, C, dist] in self.result if accepted])
        logging.debug('Number of accepted values: {} {}%'.format(len(self.accepted),
                                                                 round(len(self.accepted)/self.N*100,2)))

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

            cmap = plt.cm.jet  # define the colormap
            cmaplist = [cmap(i) for i in range(cmap.N)]  # extract all colors from the .jet map
            cmaplist[0] = ('white')  # force the first color entry to be grey
            cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

            fig = plt.figure(figsize=(9, 9))
            for i in range(self.num_of_params):
                for j in range(self.num_of_params):
                    if i == j:
                        x, y = utils.pdf_from_array_with_x(self.accepted[:, i], bins=60, range=C_limits[i])
                        ax = plt.subplot2grid((self.num_of_params, self.num_of_params), (i, i))
                        plt.hist(self.accepted[:, i], bins=num_bin_joint, normed=1, alpha=0.5, color='grey', range=C_limits[i])
                        plt.plot(x, y)
                        ax.axis(xmin=C_limits[i,0], xmax=C_limits[i, 1])
                        ax.set_xlabel(params_names[i])
                    elif i < j:
                        ax = plt.subplot2grid((self.num_of_params, self.num_of_params), (i, j))
                        ax.axis(xmin=C_limits[j, 0], xmax=C_limits[j, 1], ymin=C_limits[i, 0], ymax=C_limits[i, 1])
                        plt.hist2d(self.accepted[:, j], self.accepted[:, i], bins=num_bin_joint, normed=1, cmap=cmap, range=[C_limits[j], C_limits[i]])
            fig.tight_layout()
            # plt.legend(loc=0)
            plt.show()
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
            if scale == 'LES':
                x, y = utils.pdf_from_array_with_x(g.LES.tau_true[key].flatten(), 100, domain)
            if scale == 'TEST':
                x, y = utils.pdf_from_array_with_x(g.TEST.tau_true[key].flatten(), 100, domain)
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


def distance_between_pdf_KL(pdf_modeled, key):
    """Calculate statistical distance between two pdf as
    the Kullback-Leibler (KL) divergence (no symmetry).
    :param pdf_modeled: array of modeled pdf
    :return:            scalar of calculated distance
    """
    log_modeled = np.log(pdf_modeled, out=np.empty_like(pdf_modeled).fill(-20), where=pdf_modeled != 0)
    dist = np.sum(np.multiply(pdf_modeled, (log_modeled - g.TEST_sp.log_tau_pdf_true[key])))
    return dist

def distance_between_pdf_L1log(pdf_modeled, key):
    """Calculate statistical distance between two pdf as
    the Kullback-Leibler (KL) divergence (no symmetry).
    :param pdf_modeled: array of modeled pdf
    :return:            scalar of calculated distance
    """
    log_modeled = np.log(pdf_modeled, out=np.empty_like(pdf_modeled).fill(-20), where=pdf_modeled != 0)
    dist = 0.5*np.sum(np.abs(log_modeled - g.TEST_sp.log_tau_pdf_true[key]))
    return dist

def distance_between_pdf_L2(pdf_modeled, key):
    """Calculate statistical distance between two pdf as
    the Kullback-Leibler (KL) divergence (no symmetry).
    :param pdf_modeled: array of modeled pdf
    :return:            scalar of calculated distance
    """
    dist = np.mean((pdf_modeled - g.TEST_sp.tau_pdf_true[key])**2)
    return dist

def distance_between_pdf_L2log(pdf_modeled, key):
    """Calculate statistical distance between two pdf as
    the Kullback-Leibler (KL) divergence (no symmetry).
    :param pdf_modeled: array of modeled pdf
    :return:            scalar of calculated distance
    """
    log_modeled = np.log(pdf_modeled, out=np.empty_like(pdf_modeled).fill(-20), where=pdf_modeled != 0)
    dist = np.mean((log_modeled - g.TEST_sp.log_tau_pdf_true[key])**2)
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
        dist += distance_between_pdf_L2log(pdf_modeled=pdf, key=key)
    if dist <= g.eps:
        return [True, C, dist]
    else:
        return [False, C, dist]
