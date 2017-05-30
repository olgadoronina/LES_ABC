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
            if np.isnan(np.sum(S[i + j])):
                print('S_' + i + j + ': nan is detected ')
    return S


def strain_mod(strain):
    S_mod_sqr = 0
    for i in ['u', 'v', 'w']:
        for j in ['u', 'v', 'w']:
            S_mod_sqr += 2*np.multiply(strain[i + j], strain[i + j])
    S_mod = np.sqrt(S_mod_sqr)
    return S_mod


def scalar_product(array1, array2):
    res = 0
    for i in ['u', 'v', 'w']:
        for j in ['u', 'v', 'w']:
            res += np.multiply(array1[i + j], array2[i + j])
    return res

def Reynolds_stresses(field, Smag = None):
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


def Smagorinsky_constant_from_DNS(field, S_ij):
    tau = Reynolds_stresses(field)
    eps = -np.mean(scalar_product(tau, S_ij))
    denominator = np.mean(LES_delta**2*strain_mod(S_ij)**3)
    C_s = sqrt(eps/denominator)
    print('C_s from DNS: ', C_s)
    return C_s


def Smagorinsky_constant_dynamic(LES, TEST, S_LES, S_TEST):
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
