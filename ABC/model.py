import logging

import global_var as g
import numpy as np
import utils


class NonlinearModel(object):
    def __init__(self, data, model_params, abc_algorithm, algorithm, C_limits):

        self.M = data.field['uu'].shape[0]
        self.C_limits = C_limits
        self.N_each = algorithm['N_each']
        self.N_params = model_params['N_params']
        self.S_mod = self.calc_strain_mod(data)
        if model_params['homogeneous']:
            self.elements_in_tensor = ['uu', 'uv', 'uw', 'vv', 'vw', 'ww']
        else:
            self.elements_in_tensor = ['uu', 'uv', 'uw', 'vu', 'vv', 'vw', 'wu', 'wv', 'ww']

        self.Tensor = dict()
        for i in range(self.N_params):
            self.Tensor[str(i)] = self.calc_tensor(data, number=i)

        if abc_algorithm in ['acc-rej', 'IMCMC']:
            self.N_params_in_task = algorithm['N_params_in_task']
        else:
            self.N_params_in_task = 0

        if self.N_params_in_task == 0:
            self.sigma_from_C = self.sigma_from_C
        elif self.N_params_in_task == 1:
            self.sigma_from_C = self.sigma_from_C_1param
        elif self.N_params_in_task == 2:
            self.sigma_from_C = self.sigma_from_C_2param
        else:
            self.Reynolds_stresses_from_C = self.sigma_from_C_2param
            logging.warning('{} parameters in one task is not supported. Using 2 parameters instead'.format(
                self.N_params_in_task))
        logging.info('\n')
        logging.info('Nonlinear model with {}'.format(self.sigma_from_C.__name__))

    def calc_strain_mod(self, data):
        """Calculate module of strain tensor as |S| = (2S_ijS_ij)^1/2
        :return:       array of |S| in each point of domain
        """
        S_mod_sqr = 0
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                S_mod_sqr += 2 * np.multiply(data.S[i + j], data.S[i + j])
        return np.sqrt(S_mod_sqr)

    def calc_tensor(self, data, number):
        """Calculate tensor T_i for nonlinear viscosity model

        :param data: data class object (sparse data)
        :param number: index of tensor
        :return:  dictionary of tensor
        """

        if number == 0:
            # Calculate tensor Delta^2|S|S_ij for given field
            tensor = dict()
            for i in ['u', 'v', 'w']:
                for j in ['u', 'v', 'w']:
                    tensor[i + j] = np.multiply(self.S_mod, data.S[i + j])
            for key, value in tensor.items():
                value *= data.delta ** 2
            return tensor

        elif number == 1:
            # Calculate tensor Delta^2*(S_ikR_kj - R_ikS_kj)  for given field
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

        elif number == 2:
            # Calculate tensor Delta^2*(S_ikS_kj - 1/3{S_ikS_ki}delta_ij) for given field
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

        elif number == 3:
            # Calculate tensor Delta^2(R_ikR_kj - 1/3{R_ikR_ki}delta_ij) for given field
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

        elif number == 4:
            # Calculate tensor Delta^2/S_mod (R_ikS_klSlj - S_ikS_klRlj) for given field
            tensor1 = dict()
            for i in ['u', 'v', 'w']:
                for j in ['u', 'v', 'w']:
                    tensor1[i + j] = 0
                    tensor2 = 0
                    for k in ['u', 'v', 'w']:
                        for l in ['u', 'v', 'w']:
                            tensor1[i + j] += data.R[i + k] * data.S[k + l] * data.S[l + j]
                            tensor2 += data.S[i + k] * data.S[k + l] * data.R[l + j]
                    tensor1[i + j] -= tensor2
                    tensor1[i + j] *= data.delta ** 2
                    tensor1[i + j] /= self.S_mod
            return tensor1

        elif number == 5:
            # Calculate tensor Delta^2/S_mod (R_ikR_klSlj + S_ikR_klRlj - 2/3 {S_ikR_klRli}*delta_ij) for given field

            tensor1 = dict()
            S_R_R_inv = 0
            for i in ['u', 'v', 'w']:
                for k in ['u', 'v', 'w']:
                    for l in ['u', 'v', 'w']:
                        S_R_R_inv += data.S[i + k] * data.R[k + l] * data.R[l + i]
            for i in ['u', 'v', 'w']:
                for j in ['u', 'v', 'w']:
                    tensor1[i + j] = 0
                    tensor2 = 0
                    for k in ['u', 'v', 'w']:
                        for l in ['u', 'v', 'w']:
                            tensor1[i + j] += data.R[i + k] * data.R[k + l] * data.S[l + j]
                            tensor2 += data.S[i + k] * data.R[k + l] * data.R[l + j]
                    tensor1[i + j] += tensor2
                    if i == j:
                        tensor1[i + j] -= 2 / 3 * S_R_R_inv
                    tensor1[i + j] *= data.delta ** 2
                    tensor1[i + j] /= self.S_mod
            return tensor1

        elif number == 6:
            # Calculate tensor Delta^2/S_mod^2 (R_ikS_klR_lm_Rmj - R_ikR_klS_lmR_mj) for given field
            tensor1 = dict()
            for i in ['u', 'v', 'w']:
                for j in ['u', 'v', 'w']:
                    tensor1[i + j] = 0
                    tensor2 = 0
                    for k in ['u', 'v', 'w']:
                        for l in ['u', 'v', 'w']:
                            for m in ['u', 'v', 'w']:
                                tensor1[i + j] += data.R[i + k] * data.S[k + l] * data.R[l + m] * data.R[m + j]
                                tensor2 += data.R[i + k] * data.R[k + l] * data.S[l + m] * data.R[m + j]
                    tensor1[i + j] -= tensor2
                    tensor1[i + j] *= data.delta ** 2
                    tensor1[i + j] /= self.S_mod ** 2
            return tensor1

        elif number == 7:
            # Calculate tensor Delta^2/S_mod^2 (S_ikR_klS_lm_Smj - S_ikS_klR_lmS_mj)  for given field

            tensor1 = dict()
            for i in ['u', 'v', 'w']:
                for j in ['u', 'v', 'w']:
                    tensor1[i + j] = 0
                    tensor2 = 0
                    for k in ['u', 'v', 'w']:
                        for l in ['u', 'v', 'w']:
                            for m in ['u', 'v', 'w']:
                                tensor1[i + j] += data.S[i + k] * data.R[k + l] * data.S[l + m] * data.S[m + j]
                                tensor2 += data.S[i + k] * data.S[k + l] * data.R[l + m] * data.S[m + j]
                    tensor1[i + j] -= tensor2
                    tensor1[i + j] *= data.delta ** 2
                    tensor1[i + j] /= self.S_mod ** 2
            return tensor1

        elif number == 8:
            # Calculate tensor Delta^2/S_mod^2 (R^2S^2 + S^2R^2 - 2/3{S^2R^2}*delta_ij)  for given field

            tensor1 = dict()
            S2_R2_inv = 0
            for i in ['u', 'v', 'w']:
                for k in ['u', 'v', 'w']:
                    for l in ['u', 'v', 'w']:
                        for m in ['u', 'v', 'w']:
                            S2_R2_inv += data.S[i + k] * data.S[k + l] * data.R[l + m] * data.R[m + i]
            for i in ['u', 'v', 'w']:
                for j in ['u', 'v', 'w']:
                    tensor1[i + j] = 0
                    tensor2 = 0
                    for k in ['u', 'v', 'w']:
                        for l in ['u', 'v', 'w']:
                            for m in ['u', 'v', 'w']:
                                tensor1[i + j] += data.R[i + k] * data.R[k + l] * data.S[l + m] * data.S[m + j]
                                tensor2 += data.S[i + k] * data.S[k + l] * data.R[l + m] * data.R[m + j]
                    tensor1[i + j] += tensor2
                    if i == j:
                        tensor1[i + j] -= 2/3*S2_R2_inv
                    tensor1[i + j] *= data.delta ** 2
                    tensor1[i + j] /= self.S_mod ** 2
            return tensor1

        elif number == 9:
            # Calculate tensor Delta^2/S_mod^3 (RS^2R^2 - R^2S^2R) for given field

            tensor1 = dict()
            for i in ['u', 'v', 'w']:
                for j in ['u', 'v', 'w']:
                    tensor1[i + j] = 0
                    tensor2 = 0
                    for k in ['u', 'v', 'w']:
                        for l in ['u', 'v', 'w']:
                            for m in ['u', 'v', 'w']:
                                for n in ['u', 'v', 'w']:
                                    tensor1[i + j] += data.R[i + k] * data.S[k + l] * \
                                                      data.S[l + m] * data.R[m + n] * data.R[n + j]
                                    tensor2 += data.R[i + k] * data.R[k + l] * \
                                               data.S[l + m] * data.S[m + n] * data.R[n + j]
                    tensor1[i + j] -= tensor2
                    tensor1[i + j] *= data.delta ** 2
                    tensor1[i + j] /= self.S_mod ** 3
            return tensor1

    ####################################################################################################################
    # sigma_from_C
    ####################################################################################################################

    def sigma_from_C(self, C):
        """Calculate deviatoric part of Reynolds stresses using eddy-viscosity model.
        :param C: list of constant parameters
        :return: dict of modeled Reynolds stresses tensor
        """
        tau = dict()
        for i in self.elements_in_tensor:
            tau[i] = np.zeros((self.M, self.M, self.M))
            for j in range(self.N_params):
                tau[i] += C[j] * self.Tensor[str(j)][i]
        return tau

    def sigma_from_C_1param(self, C, dist_func):
        """ Calculate Reynolds stresses using eddy-viscosity model with 1 parameter in task.
        :param C: list of constant parameters [C0, ..., C(n-1)]
        :param dist_func: function used to calculate statistical distance
        :return: list of accepted params with distance [[C0, ..., Cn, dist], [...], [...]]
        """
        dist = np.zeros(self.N_each)
        C_last = np.linspace(self.C_limits[self.N.params - 1, 0], self.C_limits[self.N.params - 1, 1], self.N_each)
        for i in self.elements_in_tensor:
            sigma = np.zeros(self.M ** 3)
            for j in range(self.N_params - self.N_params_in_task):
                sigma += C[j] * self.Tensor[str(j)][i].flatten()
            sigma = np.outer(np.ones(self.N_each), sigma) + \
                  np.outer(C_last, self.Tensor[str(self.N_params - 1)][i].flatten())
            pdf = utils.pdf_from_array_improved(sigma, bins=g.pdf_params['bins'], domain=g.pdf_params['domain'],
                                                N_each=self.N_each)
            dist += dist_func(pdf_modeled=pdf, key=i)

        # Check for each parameter if it is accepted
        a = [0.0] * (self.N_params + 1)  # allocate memory
        a[:(self.N_params - self.N_params_in_task)] = [c for c in C]
        result = []
        for ind, distance in enumerate(dist):
            if distance <= g.eps:
                a[-2] = C_last[ind]
                a[-1] = distance
                result.append(a[:])
        return result

    def sigma_from_C_2param(self, C, dist_func):
        """ Calculate Reynolds stresses using eddy-viscosity model with 2 parameters in task.
        :param C: list of constant parameters [C0, ..., C(n-2)]
        :param dist_func: function used to calculate statistical distance
        :return: list of accepted params with distance [[C0, ..., Cn, dist], [...], [...]]
        """
        dist = np.zeros(self.N_each ** 2)
        C_last = np.linspace(self.C_limits[-1, 0], self.C_limits[-1, 1], self.N_each)
        C_before_last = np.linspace(self.C_limits[-2, 0], self.C_limits[-2, 1], self.N_each)

        for i in self.elements_in_tensor:
            sigma = np.zeros(self.M ** 3)
            for j in range(self.N_params - self.N_params_in_task):
                sigma += C[j] * self.Tensor[str(j)][i].flatten()
            for ind, c1 in enumerate(C_before_last):
                sigma_tmp = sigma.copy()
                sigma_tmp += c1 * self.Tensor[str(self.N_params - 2)][i].flatten()
                sigma_tmp = np.outer(np.ones(self.N_each), sigma_tmp) + \
                          np.outer(C_last, self.Tensor[str(self.N_params - 1)][i].flatten())
                pdf = utils.pdf_from_array_improved(sigma_tmp, bins=g.pdf_params['bins'],
                                                    domain=g.pdf_params['domain'], N_each=self.N_each)
                dist[ind * self.N_each:(ind + 1) * self.N_each] += dist_func(pdf_modeled=pdf, key=i)

        # Check for each parameter if it is accepted
        a = [0.0] * (self.N_params + 1)  # allocate memory
        a[: -3] = [c for c in C]
        result = [0.0]*self.N_each
        for ind, distance in enumerate(dist):
            if distance <= g.eps:
                a[-3] = C_before_last[ind // self.N_each]
                a[-2] = C_last[ind % self.N_each]
                a[-1] = distance
                result[ind] = a[:]
        return result

    def sigma_from_C_calibration1(self, C, dist_func):
        """ Calculate Reynolds stresses using eddy-viscosity model with 1 parameter in task.
        :param C: list of constant parameters [C0, ..., C(n-1)]
        :param dist_func: function used to calculate statistical distance
        :return: list of accepted params with distance [[C0, ..., Cn, dist], [...], [...]]
        """
        dist = np.zeros(self.N_each)
        C_last = np.linspace(self.C_limits[-1, 0], self.C_limits[-1, 1], self.N_each)
        for i in self.elements_in_tensor:
            sigma = np.zeros(self.M ** 3)
            for j in range(self.N_params - self.N_params_in_task):
                sigma += C[j] * self.Tensor[str(j)][i].flatten()
            sigma = np.outer(np.ones(self.N_each), sigma) + \
                  np.outer(C_last, self.Tensor[str(self.N_params - 1)][i].flatten())
            pdf = utils.pdf_from_array_improved(sigma, bins=g.pdf_params['bins'],
                                                domain=g.pdf_params['domain'], N_each=self.N_each)
            dist += dist_func(pdf_modeled=pdf, key=i)

        # Check for each parameter if it is accepted
        a = [0.0] * (self.N_params + 1)  # allocate memory
        a[:-self.N_params_in_task] = [c for c in C]
        result = []
        for ind, distance in enumerate(dist):
            a[-2] = C_last[ind]
            a[-1] = distance
            result.append(a[:])
        return result

    def sigma_from_C_calibration2(self, C, dist_func):
        """ Calculate Reynolds stresses using eddy-viscosity model with 2 parameters in task.
        :param C: list of constant parameters [C0, ..., C(n-2)]
        :param dist_func: function used to calculate statistical distance
        :return: list of accepted params with distance [[C0, ..., Cn, dist], [...], [...]]
        """
        dist = np.zeros(self.N_each ** 2)
        C_last = np.linspace(self.C_limits[-1, 0], self.C_limits[-1, 1], self.N_each)
        C_before_last = np.linspace(self.C_limits[-2, 0], self.C_limits[-2, 1], self.N_each)

        for i in self.elements_in_tensor:
            tau = np.zeros(self.M ** 3)
            for j in range(self.N_params - self.N_params_in_task):
                tau += C[j] * self.Tensor[str(j)][i].flatten()
            for ind, c1 in enumerate(C_before_last):
                tau_tmp = tau.copy()
                tau_tmp += c1 * self.Tensor[str(self.N_params - 2)][i].flatten()
                tau_tmp = np.tile(tau_tmp, (self.N_each, 1)) + \
                          np.outer(C_last, self.Tensor[str(self.N_params - 1)][i].flatten())
                pdf = utils.pdf_from_array_improved(tau_tmp, bins=g.pdf_params['bins'],
                                                    domain=g.pdf_params['domain'], N_each=self.N_each)
                dist[ind * self.N_each:(ind + 1) * self.N_each] += dist_func(pdf_modeled=pdf, key=i)

        a = [0.0] * (self.N_params + 1)  # allocate memory
        a[: -3] = [c for c in C]
        result = [0.0] * self.N_each**2
        for ind, distance in enumerate(dist):
            a[-3] = C_before_last[ind // self.N_each]
            a[-2] = C_last[ind % self.N_each]
            a[-1] = distance
            result[ind] = a[:]
        return result


# ####################################################################################################################
# # Reynolds_stresses_from_C
# ####################################################################################################################
# class DynamicSmagorinskyModel(object):
#     def __init__(self):
#         self.num_of_params = 1
#         self.Tensor_1 = self.calc_tensor_1()
#
#     def scalar_product(self, tensor1, tensor2):
#         """Calculate product of two tensors as S_ijT_ij = sum(S_11T_11+S_12T_12+...)
#         :return:       array of product in each point of domain
#         """
#         res = 0
#         for i in ['u', 'v', 'w']:
#             for j in ['u', 'v', 'w']:
#                 res += np.multiply(tensor1[i + j], tensor2[i + j])
#         return res
#
#     def calculate_Cs_dynamic(self):
#         """ Calculate Smagorinsky constant using Dynamic Smagorinsky model
#         :return: scalar Smagorinsky constant
#         """
#         L = dict()
#         M = dict()
#         for k in ['uu', 'uv', 'uw', 'vv', 'vw', 'ww']:
#             i, j = k[0], k[1]
#             ## L_ij
#             tensor = np.multiply(g.LES.field[i], g.LES.field[j])
#             tensor1 = filter.filter3d_array(tensor, TEST_scale)
#             L[i + j] = tensor1 - np.multiply(g.TEST.field[i], g.TEST.field[j])
#             logging.debug("Mean of L[{}] = {}".format(i + j, np.mean(L[i + j])))
#             ## M_ij
#             tensor = np.multiply(g.TEST.S_mod, g.TEST.S[i + j])
#             tensor2 = filter.filter3d_array(self.Tensor_1[i + j], TEST_scale)
#             M[i + j] = -2 * (g.TEST.delta ** 2 * tensor - g.LES.delta ** 2 * tensor2)
#             logging.debug("Mean of M[{}] = {}".format(i + j, np.mean(M[i + j])))
#         for k in ['vu', 'wu', 'wv']:
#             L[k] = L[k[1] + k[0]]
#             M[k] = M[k[1] + k[0]]
#         trace = L['uu'] + L['vv'] + L['ww']
#         logging.info('trace = ', np.mean(trace))
#         for i in ['uu', 'vv', 'ww']:
#             L[i] -= 1 / 3 * trace
#
#         # logging.debug("Calculate C_s")
#         # M_M = np.mean(self.scalar_product(M, M))
#         # logging.info('M_M = ', M_M)
#         # L_M = np.mean(self.scalar_product(L, M))
#         # logging.info('L_M = ', M_M)
#         # C_s_sqr = L_M/M_M
#         # logging.info('Cs^2 = ', C_s_sqr)
#         # C_s = sqrt(C_s_sqr)
#         # logging.debug('C_s from Dynamic model: {}'.format(C_s))
#
#         logging.debug("Calculate C_s field")
#         M_M = self.scalar_product(M, M)
#         logging.info('M_M = ', np.mean(M_M))
#         L_M = self.scalar_product(L, M)
#         logging.info('L_M = ', np.mean(L_M))
#         C_s_sqr = np.divide(L_M, M_M)
#         logging.info('Cs^2 = ', np.mean(C_s_sqr))
#         # Ploting ############################
#         map_bounds = np.linspace(-1.5, 1.5, 20)
#         plot.imagesc([C_s_sqr[:, :, 127]], map_bounds, name='Cs', titles=[r'$C_s$'])
#         plt.show()
#         x, y = utils.pdf_from_array_with_x(C_s_sqr, 100, [-0.2, 0.2])
#         plt.plot(x, y)
#         plt.xlabel(r'C_s')
#         plt.ylabel('pdf')
#         plt.show()
#         ####################################
#         C_s = np.sqrt(np.mean(C_s_sqr))
#         logging.debug('C_s from Dynamic model: {}'.format(C_s))
#         return C_s
#
#     def calc_tensor_1(self):
#         """Calculate tensor |S|S_ij for given field
#         :return:       dictionary of tensor
#         """
#         tensor = dict()
#         for i in ['u', 'v', 'w']:
#             for j in ['u', 'v', 'w']:
#                 tensor[i + j] = np.multiply(g.LES.S_mod, g.LES.S[i + j])
#         return tensor
#
#     def Reynolds_stresses_from_Cs(self, Cs=None):
#         """Calculate Reynolds stresses using Smogarinsky model.
#         :param Cs: given scalar of Smogarinsky constant
#         :return: dict of modeled Reynolds stresses tensor
#         """
#         tau = dict()
#         if not Cs:
#             Cs = self.calculate_Cs_dynamic()
#         for i in self.elements_in_tensor:
#             tau[i] = -2 * (Cs * g.LES.delta) ** 2 * self.Tensor_1[i]
#         return tau


