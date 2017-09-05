from ABC.params import *
import ABC.filter as filter
import ABC.global_var as g
import ABC.plot as plot
import ABC.utils as utils
import matplotlib.pyplot as plt

class NonlinearModel(object):

    def __init__(self, data, order):
        if HOMOGENEOUS:
            self.elements_in_tensor = ['uu', 'uv', 'uw', 'vv', 'vw', 'ww']
        else:
            self.elements_in_tensor = ['uu', 'uv', 'uw', 'vu', 'vv', 'vw', 'wu', 'wv', 'ww']
        self.num_of_params = 1
        self.Tensor_1 = self.calc_tensor_1(data)
        self.Reynolds_stresses_from_C = self.Reynolds_stresses_from_C_Smagorinsky
        if order >= 2:
            self.num_of_params = 3
            self.Tensor_2 = self.calc_tensor_2(data)
            self.Tensor_3 = self.calc_tensor_3(data)
            if MODEL == 'Kosovic':
                logging.info('Model: Kosovic formulation')
                self.Reynolds_stresses_from_C = self.Reynolds_stresses_from_C_Kosovic
            else:
                self.Reynolds_stresses_from_C = self.Reynolds_stresses_from_C_Nonlin
                if USE_C4:
                    self.num_of_params = 4
                    self.Tensor_4 = self.calc_tensor_4(data)
        if order >= 3:
            self.num_of_params = 6
            self.Tensor_5 = dict()
            self.Tensor_6 = dict()
            logging.error('ORDER parameter should be int from 1 to 2')
        if order >= 4:
            self.num_of_params = 9
            self.Tensor_7 = dict()
            self.Tensor_8 = dict()
            self.Tensor_9 = dict()
            logging.error('ORDER parameter should be int from 1 to 3')
        if order >= 5:
            self.num_of_params = 10
            self.Tensor_10 = dict()
            logging.error('ORDER parameter should be int from 1 to 4')
        if order > 5:
            logging.error('ORDER parameter should be int from 1 to 5')
    def calc_tensor_1(self, data):
        """Calculate tensor |S|S_ij for given field
        :return:       dictionary of tensor
        """

        def calc_strain_mod():
            """Calculate module of strain tensor as |S| = (2S_ijS_ij)^1/2
            :return:       array of |S| in each point of domain
            """
            S_mod_sqr = 0
            for i in ['u', 'v', 'w']:
                for j in ['u', 'v', 'w']:
                    S_mod_sqr += 2 * np.multiply(data.S[i + j], data.S[i + j])
            return np.sqrt(S_mod_sqr)

        tensor = dict()
        S_mod = calc_strain_mod()
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                tensor[i + j] = np.multiply(S_mod, data.S[i + j])
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

    def Reynolds_stresses_from_C_Smagorinsky(self, C):
        """Calculate Reynolds stresses using Smagorinsky eddy-viscosity model with constants C.
        :param C: given list of constant parameters
        :return: dict of modeled Reynolds stresses tensor
        """
        tau = dict()
        for i in self.elements_in_tensor:
            tau[i] = -2 * C[0] ** 2 * self.Tensor_1[i]
        return tau

    def Reynolds_stresses_from_C_Kosovic(self, C):
        """Calculate Reynolds stresses using eddy-viscosity model with constants C
            in Branco Kosovic formulation.
        :param C: given list of constant parameters
        :return: dict of modeled Reynolds stresses tensor
        """
        tau = dict()
        for i in self.elements_in_tensor:
            tau[i] = -2 * C[0] ** 2 * self.Tensor_1[i] + C[1] * self.Tensor_2[i] + C[2] * self.Tensor_3[i]
        return tau


    def Reynolds_stresses_from_C_Nonlin(self, C):
        """Calculate Reynolds stresses using eddy-viscosity model with constants C.
        :param C: given list of constant parameters
        :return: dict of modeled Reynolds stresses tensor
        """
        tau = dict()
        for i in self.elements_in_tensor:
            tau[i] = -C[0]**2 * (2 * self.Tensor_1[i] + C[1] * self.Tensor_2[i] + C[2] * self.Tensor_3[i])
            if USE_C4:
                tau[i] += C[3] * self.Tensor_4[i]
        return tau


class DynamicSmagorinskyModel(object):

    def __init__(self):
        self.num_of_params = 1
        self.Tensor_1 = self.calc_tensor_1()


    def scalar_product(self, tensor1, tensor2):
        """Calculate product of two tensors as S_ijT_ij = sum(S_11T_11+S_12T_12+...)
        :return:       array of product in each point of domain
        """
        res = 0
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                res += np.multiply(tensor1[i + j], tensor2[i + j])
        return res

    def calculate_Cs_dynamic(self):
        """ Calculate Smagorinsky constant using Dynamic Smagorinsky model
        :return: scalar Smagorinsky constant
        """
        L = dict()
        M = dict()
        for k in ['uu', 'uv', 'uw', 'vv', 'vw', 'ww']:
            i, j = k[0], k[1]
            ## L_ij
            tensor = np.multiply(g.LES.field[i], g.LES.field[j])
            tensor1 = filter.filter3d_array(tensor, TEST_scale)
            L[i + j] = tensor1 - np.multiply(g.TEST.field[i], g.TEST.field[j])
            logging.debug("Mean of L[{}] = {}".format(i + j, np.mean(L[i + j])))
            ## M_ij
            tensor = np.multiply(g.TEST.S_mod, g.TEST.S[i + j])
            tensor2 = filter.filter3d_array(self.Tensor_1[i + j], TEST_scale)
            M[i + j] = -2 * (g.TEST.delta ** 2 * tensor - g.LES.delta ** 2 * tensor2)
            logging.debug("Mean of M[{}] = {}".format(i + j, np.mean(M[i + j])))
        for k in ['vu', 'wu', 'wv']:
            L[k] = L[k[1] + k[0]]
            M[k] = M[k[1] + k[0]]
        trace = L['uu'] + L['vv'] + L['ww']
        print('trace = ', np.mean(trace))
        for i in ['uu', 'vv', 'ww']:
            L[i] -= 1/3*trace

        # logging.debug("Calculate C_s")
        # M_M = np.mean(self.scalar_product(M, M))
        # print('M_M = ', M_M)
        # L_M = np.mean(self.scalar_product(L, M))
        # print('L_M = ', M_M)
        # C_s_sqr = L_M/M_M
        # print('Cs^2 = ', C_s_sqr)
        # C_s = sqrt(C_s_sqr)
        # logging.debug('C_s from Dynamic model: {}'.format(C_s))

        logging.debug("Calculate C_s field")
        M_M = self.scalar_product(M, M)
        print('M_M = ', np.mean(M_M))
        L_M = self.scalar_product(L, M)
        print('L_M = ', np.mean(L_M))
        C_s_sqr = np.divide(L_M, M_M)
        print('Cs^2 = ', np.mean(C_s_sqr))
        # Ploting ############################
        map_bounds = np.linspace(-1.5, 1.5, 20)
        plot.imagesc([C_s_sqr[:, :, 127]], map_bounds, name='Cs', titles=[r'$C_s$'])
        plt.show()
        x, y = utils.pdf_from_array_with_x(C_s_sqr, 100, [-0.2, 0.2])
        plt.plot(x, y)
        plt.xlabel(r'C_s')
        plt.ylabel('pdf')
        plt.show()
        ####################################
        C_s = np.sqrt(np.mean(C_s_sqr))
        logging.debug('C_s from Dynamic model: {}'.format(C_s))
        return C_s

    def calc_tensor_1(self):
        """Calculate tensor |S|S_ij for given field
        :return:       dictionary of tensor
        """
        tensor = dict()
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                tensor[i + j] = np.multiply(g.LES.S_mod, g.LES.S[i + j])
        return tensor

    def Reynolds_stresses_from_Cs(self, Cs=None):
        """Calculate Reynolds stresses using Smogarinsky model.
        :param Cs: given scalar of Smogarinsky constant
        :return: dict of modeled Reynolds stresses tensor
        """
        tau = dict()
        if not Cs:
            Cs = self.calculate_Cs_dynamic()
        for i in self.elements_in_tensor:
            tau[i] = -2 * (Cs*g.LES.delta)**2 * self.Tensor_1[i]
        return tau