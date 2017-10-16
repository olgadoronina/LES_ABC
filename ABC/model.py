from ABC.params import *
import ABC.filter as filter
import ABC.global_var as g
import ABC.plot as plot
import ABC.utils as utils
import matplotlib.pyplot as plt
import timeit

class NonlinearModel(object):

    def __init__(self, data, order):

        self.order = order
        num_param = {'1': 1, '2': 4, '3': 6, '4': 9, '5': 10}
        self.num_of_params = num_param[str(order)]
        if order == 2 and USE_C3 == 0:
            self.num_of_params = 2
        logging.debug('Number of parameters = '+str(self.num_of_params))
        if HOMOGENEOUS:
            self.elements_in_tensor = ['uu', 'uv', 'uw', 'vv', 'vw', 'ww']
        else:
            self.elements_in_tensor = ['uu', 'uv', 'uw', 'vu', 'vv', 'vw', 'wu', 'wv', 'ww']

        self.S_mod = self.calc_strain_mod(data)
        self.Tensor = dict()
        for i in range(self.num_of_params):
            self.Tensor[str(i)] = self.calc_tensor(data, number=i)

        if self.num_of_params == 1:
            self.Reynolds_stresses_from_C = self.Reynolds_stresses_from_C_Smagorinsky
        elif N_params_in_task == 0:
            self.Reynolds_stresses_from_C = self.Reynolds_stresses_from_C_tau
        elif N_params_in_task == 1:
            self.Reynolds_stresses_from_C = self.Reynolds_stresses_from_C_Nonlin
        else:
            self.Reynolds_stresses_from_C = self.Reynolds_stresses_from_C_Nonlin_2_param2
            if N_params_in_task > 2 or N_params_in_task < 0:
                logging.warning(str(N_params_in_task) + ' parameters in one task is not supported.' +
                                'Using 2 parameters instead')
        print(str(self.Reynolds_stresses_from_C))

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
        # Calculate tensor |S|S_ij for given field
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
        # Calculate tensor (R_ikR_kj - 1/3{R_ikR_ki}delta_ij) for given field

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
        # Calculate tensor (R_ikS_klSlj - S_ikS_klRlj) for given field

            tensor1 = dict()
            for i in ['u', 'v', 'w']:
                for j in ['u', 'v', 'w']:
                    tensor1[i + j] = 0
                    tensor2 = 0
                    for k in ['u', 'v', 'w']:
                        for l in ['u', 'v', 'w']:
                            tensor1[i + j] += data.R[i + k]*data.S[k + l]*data.S[l + j]
                            tensor2 += data.S[i + k] * data.S[k + l] * data.R[l + j]
                    tensor1[i + j] -= tensor2
                    tensor1[i + j] *= data.delta ** 2
                    tensor1[i + j] /= self.S_mod
            return tensor1

        elif number == 5:
        # Calculate tensor (R_ikR_klSlj + S_ikR_klRlj - 2/3 {S_ikR_klRli}*delta_ij) for given field

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
                            tensor1[i + j] += data.R[i + k]*data.R[k + l]*data.S[l + j]
                            tensor2 += data.S[i + k] * data.R[k + l] * data.R[l + j]
                    tensor1[i + j] += tensor2
                    if i == j:
                        tensor1[i + j] -= 2 / 3 * S_R_R_inv
                    tensor1[i + j] *= data.delta ** 2
                    tensor1[i + j] /= self.S_mod
            return tensor1

        elif number == 6:
        # Calculate tensor (R_ikS_klR_lm_Rmj - R_ikR_klS_lmR_mj) for given field

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
                    tensor1[i + j] /= self.S_mod
            return tensor1

        elif number == 7:
        # Calculate tensor (S_ikR_klS_lm_Smj - S_ikS_klR_lmS_mj)  for given field

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
                    tensor1[i + j] /= self.S_mod**2
            return tensor1

        elif number == 8:
        # Calculate tensor (R^2S^2 + S^2R^2 - 2/3{S^2R^2}*delta_ij)  for given field

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
                    tensor2= 0
                    for k in ['u', 'v', 'w']:
                        for l in ['u', 'v', 'w']:
                            for m in ['u', 'v', 'w']:
                                tensor1[i + j] += data.R[i + k] * data.R[k + l] * data.S[l + m] * data.S[m + j]
                                tensor2 += data.S[i + k] * data.S[k + l] * data.R[l + m] * data.R[m + j]
                    tensor1[i + j] += tensor2
                    tensor1[i + j] *= data.delta ** 2
                    tensor1[i + j] /= self.S_mod ** 2
            return tensor1

        elif number == 9:
        # Calculate tensor (RS^2R^2 - R^2S^2R) for given field

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
    # Reynolds_stresses_from_C
    ####################################################################################################################

    def Reynolds_stresses_from_C_tau(self, C):

        """Calculate Reynolds stresses using eddy-viscosity model.
        :param C: list of constant parameters
        :return: dict of modeled Reynolds stresses tensor
        """
        tau = dict()
        for i in self.elements_in_tensor:
            tau[i] = np.zeros((M, M, M))
            for j in range(N_params):
                tau[i] += C[j] * self.Tensor[str(j)][i]
        return tau

    def Reynolds_stresses_from_C_Smagorinsky(self, C):
        """Calculate Reynolds stresses using Smagorinsky eddy-viscosity model with constants C.
        :param C: list of constant parameters
        :return: dict of modeled Reynolds stresses tensor
        """
        tau = dict()
        for i in self.elements_in_tensor:
            tau[i] = C[0] * self.Tensor['0'][i]
        return tau

    # def Reynolds_stresses_from_C_Nonlin_2_improved_step(self, C, C2, dist_func):
    #     """Calculate Reynolds stresses using eddy-viscosity model of 2nd order with 3 parameters.
    #     :param C: given list of constant parameters
    #     :return: dict of modeled Reynolds stresses tensor
    #     """
    #     dist = np.zeros(N_each)
    #     for i in self.elements_in_tensor:
    #         tau = C[0] * self.Tensor_0[i].flatten() + C[1] * self.Tensor_1[i].flatten()
    #         tensor = self.Tensor_2[i].flatten()
    #         pdf = 0
    #         for k in (range(0, N_total, step)):
    #             # start = time()
    #             tau_tmp = np.outer(np.ones(N_each), tau[k:min(k+step, N_total)]) + \
    #                       np.outer(C2, tensor[k:min(k+step, N_total)])
    #             pdf += utils.pdf_from_array_improved(tau_tmp, bins=bins, domain=domain)
    #         dist += dist_func(pdf_modeled=pdf, key=i)
    #
    #     return dist

    def Reynolds_stresses_from_C_Nonlin(self, C, dist_func):
        """ Calculate Reynolds stresses using eddy-viscosity model with 1 parameter in task.
        :param C: list of constant parameters [C0, ..., C(n-1)]
        :param dist_func: function used to calculate statistical distance
        :return: list of accepted params with distance [[C0, ..., Cn, dist], [...], [...]]
        """
        dist = np.zeros(N_each)
        C_last = utils.uniform_grid(N_params-1)
        for i in self.elements_in_tensor:
            # pr.enable()
            tau = np.zeros(M**3)
            for j in range(N_params-N_params_in_task):
                tau += C[j] * self.Tensor[str(j)][i].flatten()
            tau = np.outer(np.ones(N_each), tau)
            tau += np.outer(C_last, self.Tensor[str(N_params-1)][i].flatten())
            pdf = utils.pdf_from_array_improved(tau, bins=bins, domain=domain)
            dist += dist_func(pdf_modeled=pdf, key=i)
        # Check for each parameter if it is accepted
        a = [0.0] * (N_params + 1)  # allocate memory
        a[:(N_params - N_params_in_task)] = [c for c in C]
        result = []
        for ind, distance in enumerate(dist):
            # if distance <= g.eps:
            a[-2] = C_last[ind]
            a[-1] = distance
            result.append(a[:])
        return result

    def Reynolds_stresses_from_C_Nonlin_2_param2(self, C, dist_func):
        """ Calculate Reynolds stresses using eddy-viscosity model with 2 parameters in task.
        :param C: list of constant parameters [C0, ..., C(n-2)]
        :param dist_func: function used to calculate statistical distance
        :return: list of accepted params with distance [[C0, ..., Cn, dist], [...], [...]]
        """
        dist = np.zeros((N_each, N_each))
        C_last = utils.uniform_grid(N_params-1)
        C_before_last = utils.uniform_grid(N_params-2)

        for ind, c1 in enumerate(C_before_last):
            for i in self.elements_in_tensor:
                # pr.enable()
                tau = np.zeros(M ** 3)
                for j in range(N_params - N_params_in_task):
                    tau += C[j] * self.Tensor[str(j)][i].flatten()
                tau += c1 * self.Tensor[str(N_params-2)][i].flatten()
                tau = np.outer(np.ones(N_each), tau) + np.outer(C_last, self.Tensor[str(N_params - 21)][i].flatten())
                pdf = utils.pdf_from_array_improved(tau, bins=bins, domain=domain)
                dist += dist_func(pdf_modeled=pdf, key=i)

        # Check for each parameter if it is accepted
        a = [0.0]*(N_params+1)   # allocate memory
        a[:(N_params - N_params_in_task)] = [c for c in C]
        result = []
        for ind, distance in enumerate(dist):
            if distance <= g.eps:
                a[-3] = C_before_last[ind // N_each]
                a[-2] = C_last[ind % N_each]
                a[-1] = distance
                result.append(a[:])
        return result



####################################################################################################################
# Reynolds_stresses_from_C
####################################################################################################################
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


# def distance_between_pdf_KL(pdf_modeled, key):
#     """Calculate statistical distance between two pdf as
#     the Kullback-Leibler (KL) divergence (no symmetry).
#     :param pdf_modeled: array of modeled pdf
#     :return:            scalar of calculated distance
#     """
#     log_modeled = np.log(pdf_modeled, out=np.empty_like(pdf_modeled).fill(-20), where=pdf_modeled != 0)
#     dist = np.sum(np.multiply(pdf_modeled, (log_modeled - g.TEST_sp.log_tau_pdf_true[key])))
#     return dist
#
# def distance_between_pdf_L1log(pdf_modeled, key):
#     """Calculate statistical distance between two pdf as
#     the Kullback-Leibler (KL) divergence (no symmetry).
#     :param pdf_modeled: array of modeled pdf
#     :return:            scalar of calculated distance
#     """
#     log_modeled = np.log(pdf_modeled, out=np.empty_like(pdf_modeled).fill(-20), where=pdf_modeled != 0)
#     dist = 0.5*np.sum(np.abs(log_modeled - g.TEST_sp.log_tau_pdf_true[key]))
#     return dist
#
# def distance_between_pdf_L2(pdf_modeled, key):
#     """Calculate statistical distance between two pdf as
#     the Kullback-Leibler (KL) divergence (no symmetry).
#     :param pdf_modeled: array of modeled pdf
#     :return:            scalar of calculated distance
#     """
#     dist = np.mean((pdf_modeled - g.TEST_sp.tau_pdf_true[key])**2)
#     return dist
#
# def distance_between_pdf_L2log(pdf_modeled, key):
#     """Calculate statistical distance between two pdf.
#     :param pdf_modeled: array of modeled pdf
#     :return:            scalar of calculated distance
#     """
#     log_modeled = np.log(pdf_modeled, out=np.empty_like(pdf_modeled).fill(-20), where=pdf_modeled != 0)
#     dist = np.mean((log_modeled - g.TEST_sp.log_tau_pdf_true[key])**2, axis=1)
#     # dist = np.mean((log_modeled - g.TEST_sp.log_tau_pdf_true[key]) ** 2)
#     return dist