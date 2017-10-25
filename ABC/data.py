from ABC.params import *
import ABC.utils as utils

class Data(object):

    def __init__(self, data_dict, delta=0):
        self.field = data_dict
        self.delta = delta
        self.tau_true = self.Reynolds_stresses_from_DNS()
        self.S = None
        self.S_mod = None
        self.R = None
        self.calc_stresses()

    def calc_stresses(self):
        self.S = self.calc_strain_tensor()
        self.S_mod = self.calc_strain_mod()
        if ORDER > 1:
            self.R = self.calc_rotation_tensor()

    def field_gradient(self):
        """Calculate tensor of gradients of self.field.
        :return:      dictionary of gradient tensor
        """
        grad = dict()
        dx = np.divide(lx, N_points)
        grad['uu'], grad['uv'], grad['uw'] = np.gradient(self.field['u'], dx[0], dx[1], dx[2], edge_order=2)
        grad['vu'], grad['vv'], grad['vw'] = np.gradient(self.field['v'], dx[0], dx[1], dx[2], edge_order=2)
        grad['wu'], grad['wv'], grad['ww'] = np.gradient(self.field['w'], dx[0], dx[1], dx[2], edge_order=2)
        return grad

    def calc_strain_tensor(self):
        """Calculate strain tensor S_ij = (du_i/dx_j+du_j/dx_i) of given field.
        :return:      dictionary of strain tensor
        """
        A = self.field_gradient()
        tensor = dict()
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                tensor[i + j] = 0.5 * (A[i + j] + A[j + i])
        return tensor

    def calc_rotation_tensor(self):
        """Calculate rotation tensor R_ij = (du_i/dx_j-du_j/dx_i) of given field.
        :return:       dictionary of rotation tensor
        """
        A = self.field_gradient()
        tensor = dict()
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                tensor[i + j] = 0.5 * (A[i + j] - A[j + i])
        return tensor

    def calc_strain_mod(self):
        """Calculate module of strain tensor as |S| = (2S_ijS_ij)^1/2
        :return:       array of |S| in each point of domain
        """
        S_mod_sqr = 0
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                S_mod_sqr += 2*np.multiply(self.S[i + j], self.S[i + j])
        return np.sqrt(S_mod_sqr)

    def Reynolds_stresses_from_DNS(self):
        """Calculate Reynolds stresses using DNS data.
        :return:     dictionary of Reynolds stresses tensor
        """
        tensor = dict()
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                tensor[i + j] = self.field[i + j] - np.multiply(self.field[i], self.field[j])

        trace = tensor['uu'] + tensor['vv'] + tensor['ww']
        for i in ['uu', 'vv', 'ww']:
            tensor[i] -= trace

        return tensor


class DataSparse(object):

    def __init__(self, data, n_points):
        logging.info('Sparse data')
        self.M = n_points
        self.delta = data.delta

        # Sparse data
        self.field = self.sparse_dict(data.field)
        self.S = self.sparse_dict(data.S)
        if ORDER > 1:
            self.R = self.sparse_dict(data.R)

        # True pdf for distance calculation
        tau_true = data.Reynolds_stresses_from_DNS()
        self.tau_true = tau_true
        self.tau_pdf_true = dict()
        self.log_tau_pdf_true = dict()
        for key, value in tau_true.items():
            self.tau_pdf_true[key] = utils.pdf_from_array(value, bins, domain)
            self.log_tau_pdf_true[key] = np.log(self.tau_pdf_true[key],
                                                out=np.empty_like(self.tau_pdf_true[key]).fill(TINY_log),
                                                where=self.tau_pdf_true[key] > TINY)
        logging.info('Training data shape is ' + str(self.S['uu'].shape))

    def sparse_dict(self, data):

        def sparse_array(data):
            if data.shape[0] % self.M:
                print('Error: utils.sparse_dict(): Nonzero remainder')
            n_th = int(data.shape[0] / self.M)
            sparse_data = data[::n_th, ::n_th, ::n_th].copy()
            return sparse_data

        sparse = dict()
        for key, value in data.items():
            sparse[key] = sparse_array(value)
        return sparse



