import ABC.utils as utils
from ABC.params import *
import ABC.filter as filter
import ABC.plot as plot
import ABC.abc_class as abc_class
import ABC.data as data
import ABC.global_var as g


def main():
    ####################################################################################################################
    # Initial data
    ####################################################################################################################
    if LOAD:    # Load filtered data from file
        logging.info("Load LES and TEST data")
        LES_data = np.load(loadfile_LES)
        TEST_data = np.load(loadfile_TEST)
    else:       # Filter HIT data
        logging.info('Load HIT data')
        HIT_data = utils.read_data()
        utils.spectral_density([HIT_data['u'], HIT_data['v'], HIT_data['w']], 'DNS')
        for i in ['u', 'v', 'w']:
            for j in ['u', 'v', 'w']:
                HIT_data[i + j] = np.multiply(HIT_data[i], HIT_data[j])
        logging.info('Filter HIT data')
        LES_data = filter.filter3d(data=HIT_data, scale_k=LES_scale)
        TEST_data = filter.filter3d(data=HIT_data, scale_k=TEST_scale)
        utils.spectral_density([LES_data['u'], LES_data['v'], LES_data['w']], 'LES')
        utils.spectral_density([TEST_data['u'], TEST_data['v'], TEST_data['w']], 'TEST')
        logging.info('Writing files')
        np.savez(data_folder + 'LES.npz', **LES_data)
        np.savez(data_folder + 'TEST.npz', **TEST_data)

        map_bounds = np.linspace(np.min(HIT_data['v'][:, :, 127]), np.max(HIT_data['v'][:, :, 127]), 10)
        plot.imagesc([HIT_data['v'][:, :, 127], LES_data['v'][:, :, 127], TEST_data['v'][:, :, 127]],
                     map_bounds, name='compare_velocity', titles=[r'$DNS$', r'$LES$', r'$TEST$'])
        plt.show()


    logging.info('Create LES class')
    g.LES = data.Data(LES_data, LES_delta)
    logging.info('Create TEST class')
    g.TEST = data.Data(TEST_data, TEST_delta)
    del LES_data, TEST_data
    ####################################################################################################################
    # logging.info('Plotting velocity fields for DNS, LES and TEST scale')
    # map_bounds = np.linspace(np.min(g.LES.field['v'][:, :, 127]), np.max(g.LES.field['v'][:, :, 127]), 10)
    # plot.imagesc([g.LES.field['u'][:, :, 127], g.LES.field['v'][:, :, 127], g.LES.field['w'][:, :, 127]],
    #              map_bounds, name='LES', titles=[r'$u$', r'$v$', r'$w$'])
    # plt.show()
    # plot.imagesc([g.TEST.field['u'][:, :, 127], g.TEST.field['v'][:, :, 127], g.TEST.field['w'][:, :, 127]],
    #              map_bounds, name='TEST', titles=[r'$u$', r'$v$', r'$w$'])
    # plt.show()
    ####################################################################################################################

    ####################################################################################################################
    # ABC algorithm
    ####################################################################################################################
    logging.info('ABC algorithm')
    abc = abc_class.ABC(N=N_each**N_params, M=M, eps=eps, order=ORDER)
    abc.main_loop()
    np.savez('./ABC/plots/accepted.npz', calc_final_C=abc.accepted, dist=abc.dist)
    logging.info('Accepted parameters and distances saved in ./ABC/plots/accepted.npz')

    #########################
    # abc.accepted = np.load('./ABC/plots/accepted.npz')['c']
    # abc.dist = np.load('./ABC/plots/accepted.npz')['dist']
    # new_eps = 190
    # abc.accepted = abc.accepted[abc.dist < new_eps]
    # abc.dist = abc.dist[abc.dist < new_eps]
    # print('accepted {} values ({}%)'.format(len(abc.accepted), round(len(abc.accepted) / abc.N * 100, 2)))
    #########################

    abc.calc_final_C()
    # abc.plot_scatter()
    abc.plot_marginal_pdf()
    # abc.plot_compare_tau('TEST')
    # abc.plot_compare_tau('LES')


    # logging.info('Dynamic Smagorinsky')
    # SmagorinskyModel = model.DynamicSmagorinskyModel()
    # print(SmagorinskyModel.calculate_Cs_dynamic())
if __name__ == '__main__':
    main()
