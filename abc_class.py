from params import *
import global_var as g
import data
import utils
import plot
import parallel
from tqdm import tqdm

class ABC(object):

    def __init__(self, N, M):
        g.TEST_sp = data.DataSparse(g.TEST, M)
        g.Model = data.ViscosityModel(g.TEST_sp)
        self.C_array = self.form_C_array(N, g.Model.num_of_params)
        self.result = []
        self.accepted = []
        self.dist = []

    def form_C_array(self, N, n):
        C_array = []
        for i in range(N):
            C = []
            for j in range(n):
                C.append(rand.uniform(C_limits[j][0], C_limits[j][1]))
            C_array.append(C)
        return C_array

    def main_loop(self):
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
        logging.debug('Number of accepted values: '+str(len(self.accepted)))

    def plot_marginal_pdf(self):
        C1_accepted = self.accepted[:, 0]
        plot.histogram(C1_accepted, bins=20, label=r'$C_s$')
        if ORDER > 1:
            C2_accepted, C3_accepted , C4_accepted  = self.accepted[:, 1], self.accepted[:, 2], self.accepted[:, 3]
            plot.histogram(C2_accepted, bins=20, label=r'$C_2$')
            plot.histogram(C3_accepted, bins=20, label=r'$C_3$')
            plot.histogram(C4_accepted, bins=20, label=r'$C_4$')
            minim = np.argmin(self.dist)
            logging.debug('Minimum distance is in: ' + str(C1_accepted[minim]) + ' '
                          + str(C2_accepted[minim]) + ' ' + str(C3_accepted[minim]) + ' '
                          + str(C4_accepted[minim]))

    def calc_final_C(self):
        # Joint PDF
        if ORDER == 1:
            C1_accepted = self.accepted[:, 0]
            C_final = [C1_accepted[np.argmin(self.dist)]]
            logging.info('Estimated parameters: ' + str(C_final))
        else:
            C_joint = self.accepted
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
        return C_final

def distance_between_pdf(pdf_modeled, key):
    """Calculate statistical distance between two pdf as
    the Kullback-Leibler (KL) divergence (no symmetry).
    In the simple case, a KL divergence of 0 indicates that we can expect similar,
    while a KLr divergence of 1 indicates that the two distributions behave in a different manner.
    :param pdf_true:    array of expected pdf
    :param pdf_modeled: array of modeled pdf
    :return:            scalar of calculated distance
    """
    log_true = g.TEST_sp.log_tau_pdf_true[key]
    log_modeled = np.log(pdf_modeled, out=np.empty_like(pdf_modeled).fill(-20), where=pdf_modeled != 0)
    dist = np.sum(np.multiply(pdf_modeled, (log_modeled - log_true)))
    return dist

def work_function(C):
    """ Worker function for parallel regime (for pool.map from multiprocessing module)
    :param Cs: scalar value of sampled parameter
    :return:   list[Cs, dist] if accepted and None if not
    """
    tau = g.Model.Reynolds_stresses_from_C(C)
    dist = 0
    for key in g.Model.elements_in_tensor:
        pdf, edges = np.histogram(tau[key].flatten(), bins=bins, range=domain, normed=1)
        dist += distance_between_pdf(pdf_modeled=pdf, key=key)
    if dist <= g.eps:
        return [True, C, dist]
    else:
        return [False, C, dist]
