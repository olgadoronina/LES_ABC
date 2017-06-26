from params import *
import logging

class Data(object):

    def __init__(self, data_dict, delta, n_points):
        self.field = data_dict
        self.delta = delta
        # self.A = dict()
        self.S = dict()
        self.R = dict()
        self.S_mod = 0
        self.Tensor_1 = dict()
        self.Tensor_2 = dict()
        self.Tensor_3 = dict()
        self.Tensor_4 = dict()
        self.tau_true = dict()
        self.tau_pdf_true = dict()
        self.M = n_points

    def field_gradient(self):
        """Calculate tensor of gradients of given field.
        :param field: dictionary of field variables
        :return:      dictionary of gradient tensor
        """
        grad = dict()
        dx = np.divide(lx, self.M)
        grad['uu'], grad['uv'], grad['uw'] = np.gradient(self.field['u'], dx[0], dx[1], dx[2], edge_order=2)
        grad['vu'], grad['vv'], grad['vw'] = np.gradient(self.field['v'], dx[0], dx[1], dx[2], edge_order=2)
        grad['wu'], grad['wv'], grad['ww'] = np.gradient(self.field['w'], dx[0], dx[1], dx[2], edge_order=2)
        self.A = grad
        return grad

    def calc_strain_tensor(self):
        """Calculate strain tensor S_ij = (du_i/dx_j+du_j/dx_i) of given field.
        :return:      dictionary of strain tensor
        """
        A = self.field_gradient()
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                self.S[i + j] = 0.5 * (A[i + j] + A[j + i])

    def calc_rotation_tensor(self):
        """Calculate rotation tensor R_ij = (du_i/dx_j-du_j/dx_i) of given field.
        :return:       dictionary of rotation tensor
        """
        A = self.field_gradient()
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                self.R[i + j] = 0.5 * (A[i + j] - A[j + i])

    def calc_strain_mod(self):
        """Calculate module of strain tensor as |S| = (2S_ijS_ij)^1/2
        :return:       array of |S| in each point of domain
        """
        if not self.S:
            self.strain_tensor()
        S_mod_sqr = 0
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                S_mod_sqr += 2*np.multiply(self.S[i + j], self.S[i + j])
        self.S_mod = np.sqrt(S_mod_sqr)

    def calc_tensor_1(self):
        """Calculate tensor |S|S_ij for given field
        :return:       dictionary of tensor
        """
        if not self.S_mod:
            if not self.S:
                self.calc_strain_tensor()
            self.calc_strain_mod()
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                self.Tensor_1[i + j] = np.multiply(self.S_mod, self.S[i + j])

    def calc_tensor_2(self):
        """Calculate tensor (S_ikR_kj - R_ikS_kj)   for given field
        :return:       dictionary of tensor
        """
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                self.Tensor_2[i + j] = 0
                for k in ['u', 'v', 'w']:
                    self.Tensor_2[i + j] += np.multiply(self.S[i + k], self.R[k + j]) - \
                                            np.multiply(self.R[i + k], self.S[k + j])

    def calc_tensor_3(self):
        """Calculate tensor (S_ikS_kj - 1/3{S_ikS_ki}delta_ij) for given field
        :return:       dictionary of tensor
        """
        S_S_inv = 0
        for i in ['u', 'v', 'w']:
            for k in ['u', 'v', 'w']:
                S_S_inv += np.multiply(self.S[i + k], self.S[k + i])

        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                self.Tensor_3[i + j] = 0
                for k in ['u', 'v', 'w']:
                    self.Tensor_3[i + j] += np.multiply(self.S[i + k], self.S[k + j])
                    if i == j:
                        self.Tensor_3[i + j] -= 1/3*S_S_inv

    def calc_tensor_4(self):
        """Calculate tensor (R_ikR_kj - 1/3{R_ikR_ki}delta_ij) for given field
        :return:       dictionary of tensor
        """
        R_R_inv = 0
        for i in ['u', 'v', 'w']:
            for k in ['u', 'v', 'w']:
                R_R_inv += np.multiply(self.R[i + k], self.R[k + i])

        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                self.Tensor_4[i + j] = 0
                for k in ['u', 'v', 'w']:
                    self.Tensor_4[i + j] += np.multiply(self.R[i + k], self.R[k + j])
                    if i == j:
                        self.Tensor_4[i + j] -= 1/3*R_R_inv

    def Reynolds_stresses_from_DNS(self):
        """Calculate Reynolds stresses using DNS data.
        :return:     dictionary of Reynolds stresses tensor
        """
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                self.tau_true[i + j] = self.field[i + j] - np.multiply(self.field[i], self.field[j])

    def Reynolds_stresses_from_C(self, C):
        """Calculate Reynolds stresses using given Smagorinsky constant Cs
            (Smagorinsky model).
        :param C_s: given Smagorinsky constant Cs
        :return: dictionary of modeled Reynolds stresses tensor
        """
        tau = dict()
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                tau[i+j] = -2*C[0]**2*self.delta**2*self.Tensor_1[i+j]
                if len(C)>1:
                    tau[i + j] += C[1] * self.delta ** 2 * self.Tensor_2[i + j] +\
                                  C[2] * self.delta ** 2 * self.Tensor_3[i + j] + \
                                  C[3] * self.delta ** 2 * self.Tensor_4[i + j]
        if np.isnan(np.sum(tau[i + j])):
            print('tau_' + i + j + ': nan is detected ')
        return tau



