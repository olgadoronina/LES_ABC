import logging

import global_var as g
import numpy as np
import utils


class Data(object):

    def __init__(self, data_dict, delta, homogeneous, dx, info):
        self.field = data_dict
        self.dx = dx
        self.delta = delta
        self.tau_true = self.Reynolds_stresses_from_DNS()
        self.S = self.calc_strain_tensor()
        self.S_mod = self.calc_strain_mod()
        self.R = self.calc_rotation_tensor()
        self.A = None
        if info:
            self.A = self.field_gradient()
        if homogeneous:
            self.elements_in_tensor = ['uu', 'uv', 'uw', 'vv', 'vw', 'ww']
        else:
            self.elements_in_tensor = ['uu', 'uv', 'uw', 'vu', 'vv', 'vw', 'wu', 'wv', 'ww']


    def field_gradient(self):
        """Calculate tensor of gradients of self.field.
        :return:      dictionary of gradient tensor
        """
        grad = dict()
        grad['uu'], grad['uv'], grad['uw'] = np.gradient(self.field['u'], self.dx[0], self.dx[1], self.dx[2])
        grad['vu'], grad['vv'], grad['vw'] = np.gradient(self.field['v'], self.dx[0], self.dx[1], self.dx[2])
        grad['wu'], grad['wv'], grad['ww'] = np.gradient(self.field['w'], self.dx[0], self.dx[1], self.dx[2])
        return grad

    def calc_strain_tensor(self):
        """Calculate strain tensor S_ij = 1/2(du_i/dx_j+du_j/dx_i) of given field.
        :return:      dictionary of strain tensor
        """
        A = self.field_gradient()
        tensor = dict()
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                tensor[i + j] = 0.5 * (A[i + j] + A[j + i])
        return tensor

    def calc_rotation_tensor(self):
        """Calculate rotation tensor R_ij = 1/2(du_i/dx_j-du_j/dx_i) of given field.
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
        """Calculate deviatoric part of Reynolds stresses using DNS data.
            tau_ij = \tilde{u_iu_j} - \tilde{u_i}\tilde{u_j}
            sigma_ij = tau_ij - 1/3 tau_kk*delta_ij
        :return:     dictionary of Reynolds stresses tensor
        """
        tensor = dict()
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                tensor[i + j] = self.field[i + j] - np.multiply(self.field[i], self.field[j])

        trace = tensor['uu'] + tensor['vv'] + tensor['ww']
        for i in ['uu', 'vv', 'ww']:
            tensor[i] -= 1 / 3 * trace

        return tensor


class DataSparse(object):

    def __init__(self, data, n_training):

        logging.info('Sparse data')
        self.M = n_training
        self.delta = data.delta
        self.elements_in_tensor = data.elements_in_tensor

        # Sparse data
        self.field = self.sparse_dict(data.field)
        self.S = self.sparse_dict(data.S)
        self.R = self.sparse_dict(data.R)
        if data.A:
            self.A = self.sparse_dict(data.A)

        # True pdf for distance calculation
        self.tau_true = data.tau_true
        self.tau_pdf_true = dict()
        self.log_tau_pdf_true = dict()
        for key, value in self.tau_true.items():
            self.tau_pdf_true[key] = utils.pdf_from_array(value, g.bins, g.domain)
            where_array = np.array(self.tau_pdf_true[key] > g.TINY)
            a = np.empty_like(self.tau_pdf_true[key])
            a.fill(g.TINY_log)
            self.log_tau_pdf_true[key] = np.log(self.tau_pdf_true[key], out=a, where=where_array)

        logging.info('Training data shape is ' + str(self.S['uu'].shape))

    def sparse_dict(self, data_dict):

        def sparse_array(data_value):
            if data_value.shape[0] % self.M:
                logging.warning('Error: DataSparse.sparse_dict(): Nonzero remainder')
            n_th = int(data_value.shape[0] / self.M)
            sparse_data = data_value[::n_th, ::n_th, ::n_th].copy()
            return sparse_data

        sparse = dict()
        for key, value in data_dict.items():
            sparse[key] = sparse_array(value)
        return sparse



