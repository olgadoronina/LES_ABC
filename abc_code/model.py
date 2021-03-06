import logging

import abc_code.global_var as g
import numpy as np
import abc_code.utils as utils


class NonlinearModel(object):
    def __init__(self, path, load, model_params, abc, algorithm, C_limits, pdf_params, random, data=None):

        if data:
            self.M = data.S['uu'].shape[0]
        else:
            self.M = abc['num_training_points']
        self.S_full = dict()
        for i in data.S.keys():
            self.S_full[i] = data.S[i].flatten()
        self.S = self.S_full.copy()
        self.C_limits = C_limits
        self.random = random
        self.pdf_params = pdf_params
        if 'N_each' in algorithm.keys():
            self.N_each = algorithm['N_each']
        self.N_params = model_params['N_params']
        if model_params['homogeneous']:
            self.elements_in_tensor = ['uu', 'uv', 'uw', 'vv', 'vw', 'ww']
        else:
            self.elements_in_tensor = ['uu', 'uv', 'uw', 'vu', 'vv', 'vw', 'wu', 'wv', 'ww']

        self.sigma = dict()
        if abc['algorithm'] in ['acc-rej', 'IMCMC']:
            self.N_params_in_task = algorithm['N_params_in_task']
        else:
            self.N_params_in_task = 0
        if self.pdf_params['summary_statistics'] == 'sigma_pdf_log':
            if self.N_params_in_task == 0:
                self.sum_stat_from_C = self.sigma_pdf
            elif self.N_params_in_task == 1:
                self.sum_stat_from_C = self.sigma_from_C_1param
            elif self.N_params_in_task == 2:
                self.sum_stat_from_C = self.sigma_from_C_2param
            else:
                self.Reynolds_stresses_from_C = self.sigma_from_C_2param
                logging.warning('{} parameters in one task is not supported. Using 2 parameters instead'.format(
                    self.N_params_in_task))
        elif self.pdf_params['summary_statistics'] == 'production_pdf_log':
            if self.N_params_in_task == 0:
                self.sum_stat_from_C = self.production_pdf
        elif self.pdf_params['summary_statistics'] == 'both_pdf_log':
            if self.N_params_in_task == 0:
                self.sum_stat_from_C = self.both_pdf
        if load and random:
            self.Tensor = dict()
            for i in range(self.N_params):
                self.Tensor[str(i)] = np.load(path)[str(i)].item()
        else:
            self.S_mod = self.calc_strain_mod(data)
            self.Tensor = dict()
            logging.info('Calculate model tensors')
            for i in range(self.N_params):
                self.Tensor[str(i)] = self.calc_tensor(data, number=i)
                logging.info('Tensor {} , {}'.format(i, self.Tensor[str(i)].keys()))
            np.savez(path, **self.Tensor)
            del self.S_mod

        logging.info('Nonlinear model with {}'.format(self.sum_stat_from_C.__name__))

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
                    tensor[i + j] = tensor[i + j].flatten()
            for key, value in tensor.items():
                value *= data.delta ** 2
            for key in list(tensor.keys()):
                if key not in self.elements_in_tensor:
                    del tensor[key]
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
                    tensor[i + j] = tensor[i + j].flatten()
            for key, value in tensor.items():
                value *= data.delta ** 2
            for key in list(tensor.keys()):
                if key not in self.elements_in_tensor:
                    del tensor[key]
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
                    tensor[i + j] = tensor[i + j].flatten()
            for key, value in tensor.items():
                value *= data.delta ** 2
            for key in list(tensor.keys()):
                if key not in self.elements_in_tensor:
                    del tensor[key]
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
                    tensor[i + j] = tensor[i + j].flatten()
            for key, value in tensor.items():
                value *= data.delta ** 2
            for key in list(tensor.keys()):
                if key not in self.elements_in_tensor:
                    del tensor[key]
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
                    tensor1[i + j] = tensor1[i + j].flatten()
            for key in list(tensor1.keys()):
                if key not in self.elements_in_tensor:
                    del tensor1[key]
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
                    tensor1[i + j] = tensor1[i + j].flatten()
            for key in list(tensor1.keys()):
                if key not in self.elements_in_tensor:
                    del tensor1[key]
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
                    tensor1[i + j] = tensor1[i + j].flatten()
            for key in list(tensor1.keys()):
                if key not in self.elements_in_tensor:
                    del tensor1[key]
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
                    tensor1[i + j] = tensor1[i + j].flatten()
            for key in list(tensor1.keys()):
                if key not in self.elements_in_tensor:
                    del tensor1[key]
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
                    tensor1[i + j] = tensor1[i + j].flatten()
            for key in list(tensor1.keys()):
                if key not in self.elements_in_tensor:
                    del tensor1[key]
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
                    tensor1[i + j] = tensor1[i + j].flatten()
            for key in list(tensor1.keys()):
                if key not in self.elements_in_tensor:
                    del tensor1[key]
            return tensor1

    ####################################################################################################################
    # sum_stat_from_C
    ####################################################################################################################
    def sigma_pdf(self, C):
        self.sigma_field_from_C(C)
        sigma_pdf = np.empty((len(self.elements_in_tensor), self.pdf_params['bins']))
        for ind, key in enumerate(self.elements_in_tensor):
            sigma_pdf[ind] = utils.pdf_from_array(self.sigma[key], self.pdf_params['bins'], self.pdf_params['domain'])
        return sigma_pdf

    def production_pdf(self, C):
        self.sigma_field_from_C(C, 1)
        production = 0
        for key, value in self.sigma.items():
            production += self.sigma[key]*self.S[key]
        production = np.array(
            [utils.pdf_from_array(production, self.pdf_params['bins'], self.pdf_params['domain_production'])])
        return production

    def both_pdf(self, C):
        """ Create array of 7 pdfs (6 sigma pdf and 1 production pdf). """
        self.sigma_field_from_C(C, 1)
        both_pdf = np.empty((len(self.elements_in_tensor)+1, self.pdf_params['bins']))
        for ind, key in enumerate(self.elements_in_tensor):
            both_pdf[ind] = utils.pdf_from_array(self.sigma[key], self.pdf_params['bins'], self.pdf_params['domain'])
        production = 0
        for key, value in self.sigma.items():
            production += self.sigma[key] * self.S[key]
        both_pdf[-1] = np.array(
            [utils.pdf_from_array(production, self.pdf_params['bins'], self.pdf_params['domain_production'])])
        return both_pdf

    def sigma_field_from_C(self, C, S=None):
        """Calculate deviatoric part of Reynolds stresses using eddy-viscosity model.
        :param C: list of constant parameters
        :return: dict of modeled Reynolds stresses tensor
        """
        if self.random:
            ind = utils.rand_ind(self.random)
            for i in self.elements_in_tensor:
                self.sigma[i] = np.zeros(len(ind))
                for j in range(self.N_params):
                    self.sigma[i] += C[j] * self.Tensor[str(j)][i][ind]
                if S:
                    self.S[i] = self.S_full[i][ind]
        else:
            for i in self.elements_in_tensor:
                self.sigma[i] = np.zeros(self.M**3)
                for j in range(self.N_params):
                    self.sigma[i] += C[j] * self.Tensor[str(j)][i]


    def sigma_from_C_1param(self, C, dist_func):
        """ Calculate Reynolds stresses using eddy-viscosity model with 1 parameter in task.
        :param C: list of constant parameters [C0, ..., C(n-1)]
        :param dist_func: function used to calculate statistical distance
        :return: list of accepted params with distance [[C0, ..., Cn, dist], [...], [...]]
        """
        dist = np.zeros(self.N_each)
        # C_last = np.linspace(self.C_limits[self.N.params - 1, 0], self.C_limits[self.N.params - 1, 1], self.N_each)
        C_last = utils.uniform_grid(self.C_limits[self.N.params - 1], self.N_each)
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
        # C_last = np.linspace(self.C_limits[-1, 0], self.C_limits[-1, 1], self.N_each)
        C_last = utils.uniform_grid(self.C_limits[-1], self.N_each)
        # C_before_last = np.linspace(self.C_limits[-2, 0], self.C_limits[-2, 1], self.N_each)
        C_before_last = utils.uniform_grid(self.C_limits[-2], self.N_each)
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
        # C_last = np.linspace(self.C_limits[-1, 0], self.C_limits[-1, 1], self.N_each)
        C_last = utils.uniform_grid(self.C_limits[-1], self.N_each)
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
        # C_last = np.linspace(self.C_limits[-1, 0], self.C_limits[-1, 1], self.N_each)
        # C_before_last = np.linspace(self.C_limits[-2, 0], self.C_limits[-2, 1], self.N_each)
        C_last = utils.uniform_grid(self.C_limits[-1], self.N_each)
        C_before_last = utils.uniform_grid(self.C_limits[-2], self.N_each)

        for i in self.elements_in_tensor:
            self.sigma = np.zeros(self.M ** 3)
            for j in range(self.N_params - self.N_params_in_task):
                self.sigma += C[j] * self.Tensor[str(j)][i].flatten()
            for ind, c1 in enumerate(C_before_last):
                sigma_tmp = self.sigma.copy()
                sigma_tmp += c1 * self.Tensor[str(self.N_params - 2)][i].flatten()
                sigma_tmp = np.tile(sigma_tmp, (self.N_each, 1)) + \
                          np.outer(C_last, self.Tensor[str(self.N_params - 1)][i].flatten())
                pdf = utils.pdf_from_array_improved(sigma_tmp, bins=self.pdf_params['bins'],
                                                    domain=self.pdf_params['domain'], N_each=self.N_each)
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
class RansModel():
    def __init__(self):


    @staticmethod
    def rans(t, x):
        """

        :param x:
        :return:
        """
        global S, C1, C2, ce1, ce2

        k = x[0]  # turbulence kinetic energy
        e = x[1]  # dissipation rate
        a = np.array(x[2:])  # anisotropy tensor

        P = -k * np.sum(a * S)

        alf1 = P / e - 1 + C1
        alf2 = C2 - 4 / 3

        # Governing equations
        dx = np.zeros(8)
        dx[0] = P - e  # dk / dt
        dx[1] = (ce1 * P - ce2 * e) * e / k  # de / dt
        dx[2:] = -alf1 * e * a / k + alf2 * S  # d a_ij / dt
        return dx





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


