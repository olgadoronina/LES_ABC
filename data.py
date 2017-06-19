from params import *


class Data(object):

    def __init__(self, data_dict, delta, n_points):
        self.field = data_dict
        self.delta = delta
        self.S = dict()
        self.A = dict()
        self.S_mod = 0
        self.S_mod_S_ij = dict()
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

    def strain_tensor(self):
        """Calculate strain tensor of given field.
        :param field: dictionary of field variables
        :return:      dictionary of strain tensor
        """
        A = self.field_gradient()
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                self.S[i + j] = 0.5 * (A[i + j] + A[j + i])

    def strain_mod(self):
        """Calculate module of strain tensor as |S| = (2S_ijS_ij)^1/2
        :param strain: dictionary of strain tensor
        :return:       array of |S| in each point of domain
        """
        if not self.S:
            self.strain_tensor()
        S_mod_sqr = 0
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                S_mod_sqr += 2*np.multiply(self.S[i + j], self.S[i + j])
        self.S_mod = np.sqrt(S_mod_sqr)

    def strain_mod_strain_ij(self):
        """Calculate |S|S_ij product for given field
        :param field:  dictionary of data
        :return:       dictionary of product
        """
        if not self.S_mod:
            if not self.S:
                self.strain_tensor()
            self.strain_mod()
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                self.S_mod_S_ij[i + j] = np.multiply(self.S_mod, self.S[i + j])

    def Reynolds_stresses_from_DNS(self):
        """Calculate Reynolds stresses using DNS data.
        :return:     dictionary of Reynolds stresses tensor
        """
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                self.tau_true[i + j] = self.field[i + j] - np.multiply(self.field[i], self.field[j])

    def Reynolds_stresses_from_Cs(self, C_s):
        """Calculate Reynolds stresses using given Smagorinsky constant Cs
            (Smagorinsky model).
        :param C_s: given Smagorinsky constant Cs
        :return: dictionary of modeled Reynolds stresses tensor
        """
        if not self.S_mod_S_ij:
            print('calc S_mod_S_ij')
            self.strain_mod_strain_ij()
        tau = dict()
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                tau[i+j] = -2*(C_s*self.delta)**2*self.S_mod_S_ij[i+j]
                if np.isnan(np.sum(tau[i + j])):
                    print('tau_' + i + j + ': nan is detected ')
        return tau
