from params import *

class ViscosityModel(object):

    def __init__(self, data):
        if HOMOGENEOUS:
            self.elements_in_tensor = ['uu', 'uv', 'uw', 'vv', 'vw', 'ww']
        else:
            self.elements_in_tensor = ['uu', 'uv', 'uw', 'vu', 'vv', 'vw', 'wu', 'wv', 'ww']
        self.num_of_params = 1
        self.Tensor_1 = self.calc_tensor_1(data)
        if ORDER >= 2:
            self.num_of_params = 3
            self.Tensor_2 = self.calc_tensor_2(data)
            self.Tensor_3 = self.calc_tensor_3(data)
            if USE_C4:
                self.num_of_params = 4
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

    def Reynolds_stresses_from_C(self, C):
        """Calculate Reynolds stresses using eddy-viscosity model with constants C
            (Smagorinsky model if model is linear).
        :param C: given list of constant parameters
        :return: dict of modeled Reynolds stresses tensor
        """
        tau = dict()
        for i in self.elements_in_tensor:
            tau[i] = -2*C[0]**2 * self.Tensor_1[i]
            if ORDER > 1:
                tau[i] += C[1] * self.Tensor_2[i] + C[2] * self.Tensor_3[i]
                if USE_C4:
                    tau[i] += C[3] * self.Tensor_4[i]
        return tau
