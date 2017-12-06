import logging
import sys

import global_var as g
import init
import numpy as np


def main():
    logging.basicConfig(format='%(levelname)s: %(name)s: %(message)s', level=logging.DEBUG)
    # logging.basicConfig(filename='ABC_log.log', filemode='w',
    #                     format='%(levelname)s: %(name)s: %(message)s',
    #                     level=logging.DEBUG)

    logging.info('platform {}'.format(sys.platform))
    logging.info('python {}.{}.{}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
    logging.info('numpy {}'.format(np.__version__))
    logging.info('64 bit {}\n'.format(sys.maxsize > 2 ** 32))
    ####################################################################################################################
    # Initial data
    ####################################################################################################################
    initialize = init.Init()
    initialize.plotting()
    initialize.LES_TEST_data()
    initialize.TEST_sparse_data()
    initialize.model_on_sparse_TEST_data()
    initialize.parallel()

    if g.plot.plot_info:
        logging.info('Plot initial data info')
        # g.plot.vel_fields(scale='LES')
        # g.plot.vel_fields(scale='TEST')
        # g.plot.sigma_field(scale='LES')
        # g.plot.sigma_field(scale='TEST')
        # g.plot.sigma_pdf()
        g.plot.S_pdf()
        g.plot.A_compare()
        g.plot.spectra()

    ####################################################################################################################
    # logging.info('Strain tensors')
    # # g.HIT.strain_tensor()
    # g.LES.strain_tensor()
    # g.TEST.strain_tensor()
    # g.LES_sp.A = utils.sparse_dict(g.LES.A, M)
    # g.TEST_sp.A = utils.sparse_dict(g.TEST.A, M)
    # plot.A_compare(g.TEST_sp.A, axarr, titles, M=M, color=colors[k])
    ####################################################################################################################

    ####################################################################################################################
    # ABC algorithm
    ####################################################################################################################
    abc = initialize.ABC_algorithm()
    del initialize

    abc.main_loop()
    np.savez('./plots/accepted.npz', C=g.accepted, dist=g.dist)
    logging.info('Accepted parameters and distances saved in ./ABC/plots/accepted.npz')

    # ########################
    # # abc.accepted = np.load('./plots/accepted.npz')['C']
    # # abc.dist = np.load('./plots/accepted.npz')['dist']
    # # new_eps = 40
    # # abc.accepted = abc.accepted[abc.dist < new_eps]
    # # abc.dist = abc.dist[abc.dist < new_eps]
    # # logging.info('accepted {} values ({}%)'.format(len(abc.accepted), round(len(abc.accepted) / abc.N * 100, 2)))
    # #########################
    eps = g.eps
    initialize = init.InitPostProcess(eps)
    postproc = initialize.postprocessing()
    postproc.calc_final_C()
    postproc.plot_scatter()
    # abc.plot_marginal_pdf()
    postproc.plot_compare_tau('TEST_M')
    postproc.plot_compare_tau('TEST')
    postproc.plot_compare_tau('LES')


    # logging.info('Dynamic Smagorinsky')
    # SmagorinskyModel = model.DynamicSmagorinskyModel()
    # logging.info(SmagorinskyModel.calculate_Cs_dynamic())
if __name__ == '__main__':
    main()
