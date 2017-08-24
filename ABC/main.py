import ABC.utils as utils
from ABC.params import *
import ABC.filter as filter
import ABC.plot as plot
import ABC.abc_class as abc_class
import ABC.data as data
import ABC.global_var as g
import ABC.model as model

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
    # logging.info('Writing files')
    # np.savez('./data/T.npz', uu=g.TEST.field['uu'], uv=g.TEST.field['uv'], uw=g.TEST.field['uw'])
    del LES_data, TEST_data
    ####################################################################################################################
    # map_bounds = np.linspace(np.min(g.LES.field['v'][:, :, 127]), np.max(g.LES.field['v'][:, :, 127]), 10)
    # plot.imagesc([g.LES.field['u'][:, :, 127],g.LES.field['v'][:, :, 127], g.LES.field['w'][:, :, 127]], map_bounds,
    #              name='LES', titles=[r'$u$', r'$v$', r'$w$'])

    # logging.info('ABC algorithm')
    # abc = abc_class.ABC(N=N, M=M, eps=eps, order = ORDER)
    # abc.main_loop()
    # # abc.plot_scatter()
    # abc.plot_marginal_pdf()
    # abc.calc_final_C()
    # abc.plot_compare_tau('TEST')
    # abc.plot_compare_tau('LES')

    logging.info('Dynamic Smagorinsky')
    SmagorinskyModel = model.SmagorinskyModel()
    print(SmagorinskyModel.calculate_Cs_dynamic())


if __name__ == '__main__':
    main()
