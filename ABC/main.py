import logging
import sys
import numpy as np
import global_var as g
import init
import data
import model
import parallel
import abc_class


def main():
    logPath = './plots'
    logging.basicConfig(
        format="%(levelname)s: %(name)s:  %(message)s",
        handlers=[logging.FileHandler("{0}/{1}.log".format(logPath, 'ABC_log')), logging.StreamHandler()],
        level=logging.DEBUG)

    logging.info('platform {}'.format(sys.platform))
    logging.info('python {}.{}.{}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
    logging.info('numpy {}'.format(np.__version__))
    logging.info('64 bit {}\n'.format(sys.maxsize > 2 ** 32))

    ####################################################################################################################
    # Initialize data
    ####################################################################################################################
    params = init.CreateParams()
    init.LES_TEST_data(params.data, params.physical_case, params.compare_pdf)
    g.TEST_sp = data.DataSparse(g.TEST, params.abc['num_training_points'])
    g.TEST_Model = model.NonlinearModel(g.TEST_sp, params.model, params.abc['algorithm'], params.algorithm,
                                        params.C_limits)
    if params.parallel['N_proc'] > 1:
        g.par_process = parallel.Parallel(params.parallel['progressbar'], params.parallel['N_proc'])

    abc = abc_class.ABC(params.abc, params.algorithm, params.model['N_params'], params.parallel['N_proc'],
                        params.C_limits)
    ####################################################################################################################
    # ABC algorithm
    ####################################################################################################################
    abc.main_loop()
    # np.savez('./plots/accepted.npz', C=g.accepted, dist=g.dist)
    # logging.info('Accepted parameters and distances saved in ./ABC/plots/accepted.npz')


if __name__ == '__main__':
    main()
