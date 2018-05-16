import logging
import sys
import numpy as np
import global_var as g
import init


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
    # Initial data
    ####################################################################################################################
    initialize = init.Init()
    initialize.LES_TEST_data()
    initialize.TEST_sparse_data()
    initialize.model_on_sparse_TEST_data()
    initialize.parallel()

    abc = initialize.ABC_algorithm()
    del initialize
    ####################################################################################################################
    # ABC algorithm
    ####################################################################################################################
    abc.main_loop()
    np.savez('./plots/accepted.npz', C=g.accepted, dist=g.dist)
    logging.info('Accepted parameters and distances saved in ./ABC/plots/accepted.npz')


if __name__ == '__main__':
    main()
