import utils
from params import *
import filter
import plot
import calculate
import abc_function
import random as rand
import logging
import data
import global_var as g
import utils

def main():
    # HIT_data = utils.read_data()
    # print('\nTensor u_iu_j')
    # for i in ['u', 'v', 'w']:
    #     for j in ['u', 'v', 'w']:
    #         HIT_data[i+j] = np.multiply(HIT_data[i], HIT_data[j])
    # g.HIT = data.Data(HIT_data, HIT_delta)
    # del HIT_data

    # print('\nFilter data')
    # LES_data = filter.filter3d(data=HIT.field, scale_k=LES_scale)
    # TEST_data = filter.filter3d(data=HIT.field, scale_k=TEST_scale)

    fig, axarr = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(12, 4))
    titles = [r'$\widehat{\widetilde{A}}_{11}$', r'$\widehat{\widetilde{A}}_{12}$', r'$\widehat{\widetilde{A}}_{13}$']
    colors = ['b', 'g', 'r', 'k']

    k = 0
    for M in [256, 128, 64, 32]:
        logging.info("Load data")
        LES_data = np.load('./data/LES.npz')
        TEST_data = np.load('./data/TEST.npz')

        logging.info('Sparse data')
        Nx = [M, M, M]
        LES_data = utils.sparse_dict(LES_data, M)
        TEST_data = utils.sparse_dict(TEST_data, M)
        print(LES_data['uu'].shape)

        g.LES = data.Data(LES_data, LES_delta, Nx)
        g.TEST = data.Data(TEST_data, TEST_delta, Nx)
        del LES_data, TEST_data

        logging.info('Strain tensors')
        # g.HIT.strain_tensor()
        g.LES.strain_tensor()
        g.TEST.strain_tensor()


        # g.LES.strain_mod_strain_ij()
        # g.TEST.strain_mod_strain_ij()

        # plot.S_compare(g.TEST.S, axarr, titles, label=str(M), color=colors[k])

        plot.A_compare(g.TEST.A, axarr, titles, M=M, color=colors[k])
        # logging.info('Reynolds stresses')
        # # # T_TEST = np.load('./data/T.npz')
        # g.TEST.Reynolds_stresses_from_DNS()
        # g.LES.Reynolds_stresses_from_DNS()
        # # plot.tau_sp(LES.tau_true, 'tau_LES')
        #
        # logging.info('ABC algorithm')
        # Cs = abc_function.ABC(eps, N)
        # # plot.tau_compare(Cs)
        # print(M, Cs)

        g.LES = None
        g.TEST = None
        k += 1

        # map_bounds = np.linspace(np.min(LES.field['u'][:, :, 127]), np.max(LES.field['v'][:, :, 127]), 20)
        # plot.imagesc([LES.field['u'][:, :, 127], LES.field['v'][:, :, 127], LES.field['w'][:, :, 127]], map_bounds,
        #              name='LES_velocities', titles=[r'$\widetilde{u}$', r'$\widetilde{v}$', r'$\widetilde{w}$'])
        # map_bounds = np.linspace(np.min(HIT.field['u'][:, :, 127]), np.max(HIT.field['u'][:, :, 127]), 20)
        # plot.imagesc([HIT.field['u'][:, :, 127], LES.field['u'][:, :, 127], TEST.field['u'][:, :, 127]], map_bounds, 'fourier_tophat')

        # map_bounds = np.linspace(-0.2, 0.2, 10)
        # plot.imagesc([LES.tau_true['uu'][:, :, 127], LES.tau_true['uv'][:, :, 127], LES.tau_true['uw'][:, :, 127]], map_bounds,
        #              name='tau_LES', titles=[r'$\widetilde{\tau_{11}}$', r'$\widetilde{\tau_{12}}$', r'$\widetilde{\tau_{13}}$'])

        # logging.info('T_ijS_ij')
        # TS = np.load('./data/TS.npz')['TS']
        # TS = calculate.scalar_product(TEST.tau_true, TEST.S)
        # plot.TS(TS)
        #
        # # logging.info('Writing files')
        # # np.savez('./data/T.npz', uu=T_TEST['uu'], uv=T_TEST['uv'], uw=T_TEST['uw'])
        # # np.savez('./data/TS.npz' TS=TS)
        #
        # logging.info('Calculate Smagorinsky constant')
        # C_s = calculate.Smagorinsky_constant_dynamic()
        # C_s = calculate.Smagorinsky_constant_from_DNS(g.LES)
        # C_s = calculate.Smagorinsky_constant_from_DNS(g.TEST)


        # plot.tau_tau_sp(T_TEST, T_TEST_sp)
        # plot.tau_sp(T_TEST_sp)
        # T_TEST_modeled = calculate.Reynolds_stresses_from_Cs(TEST, 0.2, TEST_delta)
        # plot.tau_compare(T_TEST, T_TEST_modeled)
        # T_TEST_modeled_sp = calculate.Reynolds_stresses_from_Cs(TEST_sp, 0.2, TEST_delta)
        # plot.tau_compare(T_TEST_sp, T_TEST_modeled_sp)

        # logger = mp.log_to_stderr(logging.DEBUG)

    fig.tight_layout()
    plt.legend(loc=0)
    plt.show()
    del fig, axarr
    gc.collect()
if __name__ == '__main__':
    main()
