from params import *
import plot
import filter


def field_gradient(field):
    """Calculate tensor of gradients of given field.
    :param field: dictionary of field variables
    :return:      dictionary of gradient tensor
    """
    grad = dict()
    grad['uu'], grad['uv'], grad['uw'] = np.gradient(field['u'], dx[0], dx[1], dx[2])
    grad['vu'], grad['vv'], grad['vw'] = np.gradient(field['v'], dx[0], dx[1], dx[2])
    grad['wu'], grad['wv'], grad['ww'] = np.gradient(field['w'], dx[0], dx[1], dx[2])
    return grad


def strain_tensor(field):
    """Calculate strain tensor of given field.
    :param field: dictionary of field variables
    :return:      dictionary of strain tensor
    """
    S = dict()
    A = field_gradient(field)
    for i in ['u', 'v', 'w']:
        for j in ['u', 'v', 'w']:
            S[i + j] = 0.5 * (A[i + j] + A[j + i])
            # print('S_' + i + j, np.mean(S[i+j]))
            if np.isnan(np.sum(S[i + j])):
                print('S_' + i + j + ': nan is detected ')
    return S


def strain_mod(strain):
    """Calculate module of strain tensor as |S| = (2S_ijS_ij)^1/2
    :param strain: dictionary of strain tensor
    :return:       array of |S| in each point of domain
    """
    S_mod_sqr = 0
    for i in ['u', 'v', 'w']:
        for j in ['u', 'v', 'w']:
            S_mod_sqr += 2*np.multiply(strain[i + j], strain[i + j])
    S_mod = np.sqrt(S_mod_sqr)
    return S_mod

def strain_mod_strain_ij(field):
    S_mod_S_ij = dict()
    S = strain_tensor(field)
    S_mod = strain_mod(S)
    for i in ['u', 'v', 'w']:
        for j in ['u', 'v', 'w']:
            S_mod_S_ij[i + j] = np.multiply(S_mod, S[i + j])
    return S_mod_S_ij


def scalar_product(array1, array2):
    """Calculate product of two tensors as S_ijT_ij = sum(S_11T_11+S_12T_12+...)
    :param array1: dictionary of first tensor
    :param array2: dictionary of second tensor
    :return:       array of product in each point of domain
    """
    res = 0
    for i in ['u', 'v', 'w']:
        for j in ['u', 'v', 'w']:
            res += np.multiply(array1[i + j], array2[i + j])
    return res


def Reynolds_stresses_from_DNS(field, Smag = None):
    """Calculate Reynolds stresses using DNS data
    :param field: dictionary of filtered data for u_i and u_iu_j
    :param Smag: flag to substruct 1/3*tau_kk*delta_ij
    :return:     dictionary of Reynolds stresses tensor
    """
    tau = dict()
    for i in ['u', 'v', 'w']:
        for j in ['u', 'v', 'w']:
            tau[i + j] = field[i + j] - np.multiply(field[i], field[j])
            if np.isnan(np.sum(tau[i + j])):
                print('tau_' + i + j + ': nan is detected ')
    if Smag:
        trace = tau['uu'] + tau['vv'] + tau['ww']
        for i in ['uu', 'vv', 'ww']:
            tau[i] -= 1/3*trace
    return tau


def Reynolds_stresses_from_Cs(field, C_s, delta, S_mod_S_ij=None):
    if not S_mod_S_ij:
        print('calc S')
        S_mod_S_ij = strain_mod_strain_ij(field)
    tau = dict()
    for i in ['u', 'v', 'w']:
        for j in ['u', 'v', 'w']:
            tau[i+j] = -2*(C_s*delta)**2*S_mod_S_ij[i+j]
    return tau


def Smagorinsky_constant_from_DNS(field, S_ij, delta):
    """Calculate Smagorinsky constant using DNS data and dissipation rate
    :param field: dictionary of filtered data
    :param S_ij: dictionary of strain tensor
    :return:     scalar Smagorinsky constant
    """
    tau = Reynolds_stresses_from_DNS(field)
    eps = -np.mean(scalar_product(tau, S_ij))
    print('epstrue', eps)
    denominator = np.mean(delta**2*strain_mod(S_ij)**3)
    C_s = sqrt(eps/denominator)
    print('C_s from DNS: ', C_s)
    return C_s


def Smagorinsky_constant_dynamic(LES, TEST, S_LES, S_TEST):
    """ Calculate Smagorinsky constant using Dynamic Smagorinsky model
    :param LES: dictionary of filtered data on LES filter scale
    :param TEST: dictionary of filtered data on TEST filter scale
    :param S_LES: dictionary of strain tensor on LES filter scale
    :param S_TEST: dictionary of strain tensor on TEST filter scale
    :return: scalar Smagorinsky constant
    """
    print("\nL_ij")
    L = dict()
    for i in ['u', 'v', 'w']:
        for j in ['u', 'v', 'w']:
            tensor = np.multiply(LES[i], LES[j])
            tensor1 = filter.filter3d_array(tensor, TEST_scale)
            L[i + j] = tensor1 - np.multiply(TEST[i], TEST[j])
            if np.isnan(np.sum(L[i + j])):
                print('L_' + i + j + ': nan is detected ')
    trace = L['uu'] + L['vv'] + L['ww']
    for i in ['uu', 'vv', 'ww']:
        L[i] -= 1/3*trace

    print("\nM_ij")
    S_TEST_mod = strain_mod(S_TEST)
    S_LES_mod = strain_mod(S_LES)

    M = dict()
    for i in ['u', 'v', 'w']:
        for j in ['u', 'v', 'w']:
            tensor = np.multiply(S_TEST_mod, S_TEST[i + j])
            tensor1 = np.multiply(S_LES_mod, S_LES[i + j])
            tensor2 = filter.filter3d_array(tensor1, TEST_scale)
            M[i + j] = -2 * (TEST_delta ** 2 * tensor - LES_delta ** 2 * tensor2)

    print("\nCalculate C_s")
    M_M = np.mean(scalar_product(M, M))
    L_M = np.mean(scalar_product(L, M))
    C_s_sqr = L_M/M_M
    # plot.histogram(C_s_sqr.flatten(), 100, label='Cs')
    C_s = sqrt(C_s_sqr)
    print('C_s from Dynamic model:', C_s)
    return C_s

