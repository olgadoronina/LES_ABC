import logging
import os
import numpy as np

import abc_code.utils as utils
from abc_code import global_var as g


class Data(object):
    def __init__(self, data_dict, delta, dx, pdf_params):

        self.delta = delta
        self.S = self.calc_strain_tensor(data_dict, dx)
        self.R = self.calc_rotation_tensor(data_dict, dx)

        if pdf_params['summary_statistics'] == 'sigma_pdf_log':
            g.sum_stat_true = self.deviatoric_stresses_pdf(data_dict, pdf_params)
            production = self.production_rate_pdf(data_dict, pdf_params)
            with open(os.path.join(g.path['output'], 'sum_stat_true'), 'wb') as f:
                np.savetxt(f, [g.sum_stat_true['uu'], g.sum_stat_true['uv'], g.sum_stat_true['uw'], production])
        elif pdf_params['summary_statistics'] == 'production_pdf_log':
            sigma = self.deviatoric_stresses_pdf(data_dict, pdf_params)
            g.sum_stat_true = self.production_rate_pdf(data_dict, pdf_params)
            with open(os.path.join(g.path['output'], 'sum_stat_true'), 'wb') as f:
                np.savetxt(f, [sigma['uu'], sigma['uv'], sigma['uw'], g.sum_stat_true])
        elif pdf_params['summary_statistics'] == 'production_mean':
            g.sum_stat_true = g.production_rate_mean(data_dict)


    @staticmethod
    def field_gradient(field, dx):
        """Calculate tensor of gradients of self.field.
        :return:      dictionary of gradient tensor
        """
        grad = dict()
        grad['uu'], grad['uv'], grad['uw'] = np.gradient(field['u'], dx[0], dx[1], dx[2])
        grad['vu'], grad['vv'], grad['vw'] = np.gradient(field['v'], dx[0], dx[1], dx[2])
        grad['wu'], grad['wv'], grad['ww'] = np.gradient(field['w'], dx[0], dx[1], dx[2])
        return grad

    def calc_strain_tensor(self, field, dx):
        """Calculate strain tensor S_ij = 1/2(du_i/dx_j+du_j/dx_i) of given field.
        :return:      dictionary of strain tensor
        """
        A = self.field_gradient(field, dx)
        tensor = dict()
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                tensor[i + j] = 0.5 * (A[i + j] + A[j + i])
        return tensor

    def calc_rotation_tensor(self, field, dx):
        """Calculate rotation tensor R_ij = 1/2(du_i/dx_j-du_j/dx_i) of given field.
        :return:       dictionary of rotation tensor
        """
        A = self.field_gradient(field, dx)
        tensor = dict()
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                tensor[i + j] = 0.5 * (A[i + j] - A[j + i])
        return tensor

    def calc_tau(self, field):
        """Calculate Reynolds stresses field using DNS data.
            tau_ij = \tilde{u_iu_j} - \tilde{u_i}\tilde{u_j}
        :return:     dictionary of Reynolds stresses tensor
        """
        tau = dict()
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                tau[i + j] = field[i + j] - np.multiply(field[i], field[j])
        return tau

    def deviatoric_stresses_pdf(self, field, pdf_params):
        """Calculate pdf of deviatoric stresses using DNS data.
            sigma_ij = tau_ij - 1/3 tau_kk*delta_ij
        :return:     dictionary of log pdfs of deviatoric stresses
        """
        tau = self.calc_tau(field)
        trace = tau['uu'] + tau['vv'] + tau['ww']
        for i in ['uu', 'vv', 'ww']:
            tau[i] -= 1 / 3 * trace
        sigma = tau
        log_sigma_pdf = dict()
        for key, value in sigma.items():
            sigma_pdf = utils.pdf_from_array(value, pdf_params['bins'], pdf_params['domain'])
            log_sigma_pdf[key] = utils.take_safe_log(sigma_pdf)

        return log_sigma_pdf

    def production_rate_pdf(self, field, pdf_params):
        """Calculate kinetic energy production rate using DNS data.
            P = -\tau_ij \partial\tilde{u_i}/\partial x_j
        :return: log of pdf of production rate
        """
        tau = self.calc_tau(field)
        trace = tau['uu'] + tau['vv'] + tau['ww']
        for i in ['uu', 'vv', 'ww']:
            tau[i] -= 1 / 3 * trace
        prod_rate = 0
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                prod_rate += tau[i + j]*self.S[i + j]
        prod_rate_pdf = utils.pdf_from_array(prod_rate, pdf_params['bins'], pdf_params['domain_production'])
        log_prod_pdf = utils.take_safe_log(prod_rate_pdf)
        return log_prod_pdf

    def production_rate_mean(self, field):
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

    def __init__(self, path, load, data=None, n_training=None):

        if load:
            sparse_data = np.load(os.path.join(path, 'TEST_sp.npz'))
            self.delta = sparse_data['delta'].item()
            self.S = sparse_data['S'].item()
            self.R = sparse_data['R'].item()
            g.sum_stat_true = np.load(os.path.join(path, 'sum_stat_true.npz'))
            logging.info('Training data shape is ' + str(self.S['uu'].shape))
        else:
            M = n_training
            self.delta = data.delta
            logging.info('Sparse data')
            self.S = self.sparse_dict(data.S, M)
            self.R = self.sparse_dict(data.R, M)
            logging.info('Training data shape is ' + str(self.S['uu'].shape))
            np.savez(os.path.join(path, 'TEST_sp.npz'), delta=self.delta, S=self.S, R=self.R)
            # np.savez(os.path.join(path, 'sum_stat_true.npz'), **g.sum_stat_true)

    def sparse_array(self, data_value, M):

        if data_value.shape[0] % M:
            logging.warning('Error: DataSparse.sparse_dict(): Nonzero remainder')
        n_th = int(data_value.shape[0] / M)
        sparse_data = data_value[::n_th, ::n_th, ::n_th].copy()
        return sparse_data

    def sparse_dict(self, data_dict, M):

        sparse = dict()
        for key, value in data_dict.items():
            sparse[key] = self.sparse_array(value, M)
        return sparse



