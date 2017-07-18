from params import *
import logging

class Data(object):

    def __init__(self, data_dict, delta=0):
        self.field = data_dict
        self.delta = delta
        self.tau_true = self.Reynolds_stresses_from_DNS()
        self.S = self.calc_strain_tensor()
        self.S_mod = self.calc_strain_mod()
        if ORDER > 1:
            self.R = self.calc_rotation_tensor()

    def field_gradient(self):
        """Calculate tensor of gradients of given field.
        :param field: dictionary of field variables
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
        return tensor


class DataSparse(object):

    def __init__(self, data, n_points):
        logging.info('Sparse data')
        self.M = n_points
        self.field = self.sparse_dict(data.field)
        self.delta = data.delta
        self.S = self.sparse_dict(data.S)
        self.S_mod = self.calc_strain_mod()
        if ORDER > 1:
            self.R = self.sparse_dict(data.R)
        self.tau_true = self.Reynolds_stresses_from_DNS()
        self.tau_pdf_true = dict()
        self.log_tau_pdf_true = dict()
        for key, value in self.tau_true.items():
            self.tau_pdf_true[key] = utils.pdf_from_array(value, bins, domain)[1]
            self.log_tau_pdf_true[key] = np.log(self.tau_pdf_true[key],
                                                out=np.empty_like(self.tau_pdf_true[key]).fill(-20),
                                                where=self.tau_pdf_true[key] != 0)
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
        tau = dict()
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                tau[i + j] = self.field[i + j] - np.multiply(self.field[i], self.field[j])
        return tau

class ViscosityModel(object):

    def __init__(self, data):
        if HOMOGENEOUS:
            self.elements_in_tensor = ['uu', 'uv', 'uw', 'vv', 'vw', 'ww']
        else:
            self.elements_in_tensor = ['uu', 'uv', 'uw', 'vu', 'vv', 'vw', 'wu', 'wv', 'ww']
        self.num_of_params = 1
        self.Tensor_1 = self.calc_tensor_1(data)
        if ORDER >= 2:
            self.num_of_params = 4
            self.Tensor_2 = self.calc_tensor_2(data)
            self.Tensor_3 = self.calc_tensor_3(data)
            self.Tensor_4 = self.calc_tensor_4(data)
        if ORDER >= 3:
            self.num_of_params = 6
            self.Tensor_5 = dict()
            self.Tensor_6 = dict()
            logging.error('ORDER parameter should be int from 1 to 2')
        if ORDER >= 4:
            self.num_of_params = 9
            self.Tensor_7 = dict()
            self.Tensor_8 = dict()
            self.Tensor_9 = dict()
            logging.error('ORDER parameter should be int from 1 to 3')
        if ORDER >= 5:
            self.num_of_params = 10
            self.Tensor_10 = dict()
            logging.error('ORDER parameter should be int from 1 to 4')
        if ORDER > 5:
            logging.error('ORDER parameter should be int from 1 to 5')

    def calc_tensor_1(self, data):
        """Calculate tensor |S|S_ij for given field
        :return:       dictionary of tensor
        """
        tensor = dict()
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                tensor[i + j] = np.multiply(data.S_mod, data.S[i + j])
        for key, value in tensor.items():
            value *= data.delta ** 2
        return tensor

    def calc_tensor_2(self, data):
        """Calculate tensor (S_ikR_kj - R_ikS_kj)   for given field
        :return:       dictionary of tensor
        """
        tensor = dict()
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                tensor[i + j] = 0
                for k in ['u', 'v', 'w']:
                    tensor[i + j] += np.multiply(data.S[i + k], data.R[k + j]) - \
                                            np.multiply(data.R[i + k], data.S[k + j])
        for key, value in tensor.items():
            value *= data.delta ** 2
        return tensor

    def calc_tensor_3(self, data):
        """Calculate tensor (S_ikS_kj - 1/3{S_ikS_ki}delta_ij) for given field
        :return:       dictionary of tensor
        """
        tensor = dict()
        S_S_inv = 0
        for i in ['u', 'v', 'w']:
            for k in ['u', 'v', 'w']:
                S_S_inv += np.multiply(data.S[i + k], data.S[k + i])
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                tensor[i + j] = 0
                for k in ['u', 'v', 'w']:
                    tensor[i + j] += np.multiply(data.S[i + k], data.S[k + j])
                    if i == j:
                        tensor[i + j] -= 1 / 3 * S_S_inv
        for key, value in tensor.items():
            value *= data.delta ** 2
        return tensor

    def calc_tensor_4(self, data):
        """Calculate tensor (R_ikR_kj - 1/3{R_ikR_ki}delta_ij) for given field
        :return:       dictionary of tensor
        """
        tensor = dict()
        R_R_inv = 0
        for i in ['u', 'v', 'w']:
            for k in ['u', 'v', 'w']:
                R_R_inv += np.multiply(data.R[i + k], data.R[k + i])
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                tensor[i + j] = 0
                for k in ['u', 'v', 'w']:
                    tensor[i + j] += np.multiply(data.R[i + k], data.R[k + j])
                    if i == j:
                        tensor[i + j] -= 1 / 3 * R_R_inv
        for key, value in tensor.items():
            value *= data.delta ** 2
        return tensor

    def Reynolds_stresses_from_C(self, C):
        """Calculate Reynolds stresses using eddy-viscosity model with constants C
            (Smagorinsky model if model is linear).
        :param C: given list of constant parameters
        :return: dict of modeled Reynolds stresses tensor
        """
        tau = dict()
        for i in self.elements_in_tensor:
            tau[i] = -2*C[0]**2*self.Tensor_1[i]
            if ORDER > 1:
                tau[i] += C[1] * self.Tensor_2[i] + \
                          C[2] * self.Tensor_3[i] + \
                          C[3] * self.Tensor_4[i]
        return tau
