import plot
import filter
from params import *
import glob


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


def Smagorinsky_constant_from_DNS(data):
    """Calculate Smagorinsky constant using DNS data and dissipation rate
    :param data: reference to the data object
    :return:     scalar Smagorinsky constant
    """
    if not data.tau_true:
        data.Reynolds_stresses_from_DNS()
    if not data.S_mod:
        data.strain_mod()
    eps = -np.mean(scalar_product(data.tau_true, data.S))
    print('eps_true', eps)
    denominator = np.mean(data.delta**2*data.S_mod**3)
    C_s = sqrt(eps/denominator)
    print('C_s from DNS: ', C_s)
    return C_s


def Smagorinsky_constant_dynamic():
    """ Calculate Smagorinsky constant using Dynamic Smagorinsky model
    :return: scalar Smagorinsky constant
    """
    print("\nL_ij")
    L = dict()
    print(glob.LES)
    for i in ['u', 'v', 'w']:
        for j in ['u', 'v', 'w']:
            tensor = np.multiply(glob.LES.field[i], glob.LES.field[j])
            tensor1 = filter.filter3d_array(tensor, TEST_scale)
            L[i + j] = tensor1 - np.multiply(glob.TEST.field[i], glob.TEST.field[j])
            if np.isnan(np.sum(L[i + j])):
                print('L_' + i + j + ': nan is detected ')
    trace = L['uu'] + L['vv'] + L['ww']
    for i in ['uu', 'vv', 'ww']:
        L[i] -= 1/3*trace

    print("\nM_ij")
    if not glob.LES.S_mod:
        glob.LES.strain_mod()
    if not glob.TEST.S_mod:
        glob.TEST.strain_mod()

    M = dict()
    for i in ['u', 'v', 'w']:
        for j in ['u', 'v', 'w']:
            tensor = np.multiply(glob.TEST.S_mod, glob.TEST.S[i + j])
            tensor1 = np.multiply(glob.LES.S_mod, glob.LES.S[i + j])
            tensor2 = filter.filter3d_array(tensor1, TEST_scale)
            M[i + j] = -2 * (glob.TEST.delta ** 2 * tensor - glob.LES.delta ** 2 * tensor2)

    print("\nCalculate C_s")
    M_M = np.mean(scalar_product(M, M))
    L_M = np.mean(scalar_product(L, M))
    C_s_sqr = L_M/M_M
    C_s = sqrt(C_s_sqr)
    print('C_s from Dynamic model:', C_s)
    return C_s

