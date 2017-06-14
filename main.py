
import utils
from params import *
import filter
import plot
import calculate
import abc_function
import random as rand
import logging

def main():
    # HIT = utils.read_data()
    # print('\nTensor u_iu_j')
    # for i in ['u', 'v', 'w']:
    #     for j in ['u', 'v', 'w']:
    #         HIT[i+j] = np.multiply(HIT[i], HIT[j])
    #
    # print('\nFilter data')
    # LES = filter.filter3d(data=HIT, scale_k=LES_scale)
    # TEST = filter.filter3d(data=HIT, scale_k=TEST_scale)

    print("\nLoad data")
    LES = np.load('./data/LES.npz')
    TEST = np.load('./data/TEST.npz')

    # map_bounds = np.linspace(np.min(LES['u'][:, :, 127]), np.max(LES['v'][:, :, 127]), 20)
    # plot.imagesc([LES['u'][:, :, 127], LES['v'][:, :, 127], LES['w'][:, :, 127]], map_bounds,
    #              name='LES_velocities', titles=[r'$\widetilde{u}$', r'$\widetilde{v}$', r'$\widetilde{w}$'])
    # map_bounds = np.linspace(np.min(HIT['u'][:, :, 127]), np.max(HIT['u'][:, :, 127]), 20)
    # plot.imagesc([HIT['u'][:, :, 127], LES['u'][:, :, 127], TEST['u'][:, :, 127]], map_bounds, 'fourier_tophat')

    # print("\nStrain tensors")
    # S = calculate.strain_tensor(HIT)
    # S_LES = calculate.strain_tensor(LES)
    # S_TEST = calculate.strain_tensor(TEST)

    # print("\nReynolds stresses")
    # # T_TEST = np.load('./data/T.npz')
    T_TEST = calculate.Reynolds_stresses_from_DNS(TEST)
    # T_LES = calculate.Reynolds_stresses_from_DNS(LES)
    # plot.tau_sp(T_LES, 'tau_LES')

    # map_bounds = np.linspace(-0.2, 0.2, 10)
    # plot.imagesc([T_LES['uu'][:, :, 127], T_LES['uv'][:, :, 127], T_LES['uw'][:, :, 127]], map_bounds,
    #              name='tau_LES', titles=[r'$\widetilde{\tau_{11}}$', r'$\widetilde{\tau_{12}}$', r'$\widetilde{\tau_{13}}$'])

    # print('\nT_ijS_ij')
    # # TS = np.load('./data/TS.npz')['TS']
    # # TS = calculate.scalar_product(T_TEST, S_TEST)
    # # plot.TS(TS)
    #
    # # print('\nWriting files')
    # # np.savez('./data/T.npz', uu=T_TEST['uu'], uv=T_TEST['uv'], uw=T_TEST['uw'])
    # # np.savez('./data/TS.npz' TS=TS)
    #
    # print('\nCalculate Smagorinsky constant')
    # C_s = calculate.Smagorinsky_constant_dynamic(LES, TEST, S_LES, S_TEST)
    # C_s = calculate.Smagorinsky_constant_from_DNS(LES, S_LES, LES_delta)
    # C_s = calculate.Smagorinsky_constant_from_DNS(TEST, S_TEST, TEST_delta)

    # LES_sp = utils.sparse_dict(LES, 16)
    # TEST_sp = utils.sparse_dict(TEST, 16)
    # T_TEST_sp = utils.sparse_dict(T_TEST, 16)

    # plot.tau_tau_sp(T_TEST, T_TEST_sp)
    # plot.tau_sp(T_TEST_sp)
    # T_TEST_modeled = calculate.Reynolds_stresses_from_Cs(TEST, 0.2, TEST_delta)
    # plot.tau_compare(T_TEST, T_TEST_modeled)
    # T_TEST_modeled_sp = calculate.Reynolds_stresses_from_Cs(TEST_sp, 0.2, TEST_delta)
    # plot.tau_compare(T_TEST_sp, T_TEST_modeled_sp)

    # logger = mp.log_to_stderr(logging.DEBUG)

    Cs = abc_function.ABC(T_TEST, TEST)
    # plot.tau_compare(T_LES, calculate.Reynolds_stresses_from_Cs(LES, Cs, LES_delta))
    print(Cs)
if __name__ == '__main__':
    main()
