import utils
from params import *
import filter
import plot
import calculate
import abc_function
import random as rand
import logging
import data
import glob

def main():
    HIT_data = utils.read_data()
    print('\nTensor u_iu_j')
    for i in ['u', 'v', 'w']:
        for j in ['u', 'v', 'w']:
            HIT_data[i+j] = np.multiply(HIT_data[i], HIT_data[j])
    glob.HIT = data.Data(HIT_data, HIT_delta)
    del HIT_data

    # print('\nFilter data')
    # LES_data = filter.filter3d(data=HIT.field, scale_k=LES_scale)
    # TEST_data = filter.filter3d(data=HIT.field, scale_k=TEST_scale)
    # print("\nLoad data")
    LES_data = np.load('./data/LES.npz')
    TEST_data = np.load('./data/TEST.npz')
    glob.LES = data.Data(LES_data, LES_delta)
    glob.TEST = data.Data(TEST_data, TEST_delta)
    del LES_data, TEST_data

    # map_bounds = np.linspace(np.min(LES.field['u'][:, :, 127]), np.max(LES.field['v'][:, :, 127]), 20)
    # plot.imagesc([LES.field['u'][:, :, 127], LES.field['v'][:, :, 127], LES.field['w'][:, :, 127]], map_bounds,
    #              name='LES_velocities', titles=[r'$\widetilde{u}$', r'$\widetilde{v}$', r'$\widetilde{w}$'])
    # map_bounds = np.linspace(np.min(HIT.field['u'][:, :, 127]), np.max(HIT.field['u'][:, :, 127]), 20)
    # plot.imagesc([HIT.field['u'][:, :, 127], LES.field['u'][:, :, 127], TEST.field['u'][:, :, 127]], map_bounds, 'fourier_tophat')

    print("\nStrain tensors")
    # glob.HIT.strain_tensor()
    # glob.LES.strain_tensor()
    # glob.TEST.strain_tensor()

    print("\nReynolds stresses")
    # # T_TEST = np.load('./data/T.npz')
    # glob.TEST.Reynolds_stresses_from_DNS()
    # glob.LES.Reynolds_stresses_from_DNS()
    # plot.tau_sp(LES.tau_true, 'tau_LES')

    # map_bounds = np.linspace(-0.2, 0.2, 10)
    # plot.imagesc([LES.tau_true['uu'][:, :, 127], LES.tau_true['uv'][:, :, 127], LES.tau_true['uw'][:, :, 127]], map_bounds,
    #              name='tau_LES', titles=[r'$\widetilde{\tau_{11}}$', r'$\widetilde{\tau_{12}}$', r'$\widetilde{\tau_{13}}$'])

    print('\nT_ijS_ij')
    # TS = np.load('./data/TS.npz')['TS']
    # TS = calculate.scalar_product(TEST.tau_true, TEST.S)
    # plot.TS(TS)
    #
    # # print('\nWriting files')
    # # np.savez('./data/T.npz', uu=T_TEST['uu'], uv=T_TEST['uv'], uw=T_TEST['uw'])
    # # np.savez('./data/TS.npz' TS=TS)
    #
    print('\nCalculate Smagorinsky constant')
    # C_s = calculate.Smagorinsky_constant_dynamic()
    # C_s = calculate.Smagorinsky_constant_from_DNS(glob.LES)
    # C_s = calculate.Smagorinsky_constant_from_DNS(glob.TEST)

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

    Cs = abc_function.ABC()
    # plot.tau_compare(T_LES, calculate.Reynolds_stresses_from_Cs(LES, Cs, LES_delta))
    print(Cs)
if __name__ == '__main__':
    main()
