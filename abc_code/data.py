import logging
import os
import numpy as np

import abc_code.utils as utils
from abc_code import global_var as g



class Data(object):

    def __init__(self, data_dict, delta, dx, pdf_params):
        self.field = data_dict
        self.dx = dx
        self.delta = delta

        self.S = self.calc_strain_tensor()
        self.R = self.calc_rotation_tensor()
        if pdf_params['summary_statistics'] == 'sigma_pdf_log':
            self.sum_stat_true = self.deviatoric_stresses_pdf(pdf_params)
            np.savetxt(os.path.join(g.path['output'], 'sum_stat_true'),
                       [self.sum_stat_true['uu'], self.sum_stat_true['uv'], self.sum_stat_true['uw']])
        elif pdf_params['summary_statistics'] == 'production_pdf_log':
            self.sum_stat_true = self.production_rate_pdf(pdf_params)
            np.savetxt(os.path.join(g.path['output'], 'sum_stat_true'), self.sum_stat_true)
        elif pdf_params['summary_statistics'] == 'production_mean':
            self.sum_stat_true = self.production_rate_mean()

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
        S_mod = 0
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                S_mod += np.multiply(self.S[i + j], self.S[i + j])
        return S_mod

    def calc_tau(self):
        """Calculate Reynolds stresses field using DNS data.
            tau_ij = \tilde{u_iu_j} - \tilde{u_i}\tilde{u_j}
        :return:     dictionary of Reynolds stresses tensor
        """
        tau = dict()
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                tau[i + j] = self.field[i + j] - np.multiply(self.field[i], self.field[j])
        return tau

    def deviatoric_stresses_pdf(self, pdf_params):
        """Calculate pdf of deviatoric stresses using DNS data.
            sigma_ij = tau_ij - 1/3 tau_kk*delta_ij
        :return:     dictionary of log pdfs of deviatoric stresses
        """
        tau = self.calc_tau()
        trace = tau['uu'] + tau['vv'] + tau['ww']
        for i in ['uu', 'vv', 'ww']:
            tau[i] -= 1 / 3 * trace
        sigma = tau
        log_sigma_pdf = dict()
        for key, value in sigma.items():
            sigma_pdf = utils.pdf_from_array(value, pdf_params['bins'], pdf_params['domain'])
            log_sigma_pdf[key] = utils.take_safe_log(sigma_pdf)

        return log_sigma_pdf

    def production_rate_pdf(self, pdf_params):
        """Calculate kinetic energy production rate using DNS data.
            P = -\tau_ij \partial\tilde{u_i}/\partial x_j
        :return: log of pdf of production rate
        """
        tau = self.calc_tau()
        trace = tau['uu'] + tau['vv'] + tau['ww']
        for i in ['uu', 'vv', 'ww']:
            tau[i] -= 1 / 3 * trace
        prod_rate = 0
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                prod_rate += tau[i + j]*self.S[i + j]
        prod_rate_pdf = utils.pdf_from_array(prod_rate, pdf_params['bins'], pdf_params['domain'])
        log_prod_pdf = utils.take_safe_log(prod_rate_pdf)
        return log_prod_pdf

    def production_rate_mean(self):
        """Calculate kinetic energy production rate using DNS data.
            P = \tau_ij \tilde{S_ij}
        :return: mean of production rate (single value)
        """
        tau = self.calc_tau()
        prod_rate = 0
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                prod_rate += tau[i + j]*self.S[i + j]
        prod_rate_mean = np.mean(prod_rate)
        return prod_rate_mean


class DataSparse(object):

    def __init__(self, data, n_training):

        logging.info('Sparse data')
        self.M = n_training
        self.delta = data.delta

        # Sparse data
        self.field = self.sparse_dict(data.field)
        self.S = self.sparse_dict(data.S)
        self.R = self.sparse_dict(data.R)
        self.sum_stat_true = data.sum_stat_true
        logging.info('Training data shape is ' + str(self.S['uu'].shape))

    def sparse_array(self, data_value):

        if data_value.shape[0] % self.M:
            logging.warning('Error: DataSparse.sparse_dict(): Nonzero remainder')
        n_th = int(data_value.shape[0] / self.M)
        sparse_data = data_value[::n_th, ::n_th, ::n_th].copy()
        return sparse_data

    def sparse_dict(self, data_dict):

        sparse = dict()
        for key, value in data_dict.items():
            sparse[key] = self.sparse_array(value)
        return sparse



