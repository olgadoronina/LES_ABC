import utils
from params import *
import filter
import plot
import calculate
import abc_class
import data
import global_var as g


def main():

    if LOAD:    # Load filtered data from file
        logging.info("Load LES and TEST data")
        LES_data = np.load(loadfile_LES)
        TEST_data = np.load(loadfile_TEST)
        logging.info('Create LES class')
        g.LES = data.Data(LES_data, LES_delta)
        logging.info('Create TEST class')
        g.TEST = data.Data(TEST_data, TEST_delta)
        del LES_data, TEST_data

    else:       # Filter HIT data
        logging.info('Load HIT data')
        HIT_data = utils.read_data()
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                HIT_data[i + j] = np.multiply(HIT_data[i], HIT_data[j])
        logging.info('Filter HIT data')
        LES_data = filter.filter3d(data=HIT_data, scale_k=LES_scale)
        TEST_data = filter.filter3d(data=HIT_data, scale_k=TEST_scale)
        logging.info('Create LES class')
        g.LES = data.Data(LES_data, LES_delta)
        logging.info('Create TEST class')
        g.TEST = data.Data(TEST_data, TEST_delta)
        del LES_data, TEST_data

    # # logging.info('Plot true Reynolds stresses')
    # # plot.tau_sp(LES.tau_true, 'tau_LES')

    logging.info('ABC algorithm')
    abc = abc_class.ABC(N, M)
    abc.main_loop()
    abc.plot_marginal_pdf()
    C = abc.calc_final_C()
    #
    # # C = abc_function.ABC(N)
    # plot.tau_compare(C)
    # print(M, C)
    #
    # # g.LES_sp = None
    # g.TEST_sp = None

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

    # fig.tight_layout()
    # plt.legend(loc=0)
    # plt.show()
    # del fig, axarr
    # gc.collect()
if __name__ == '__main__':
    main()
