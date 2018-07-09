import logging
import sys
import os
import yaml
import numpy as np

import abc_code.global_var as g
import abc_code.data as data
import abc_code.model as model
import abc_code.parallel as parallel
import abc_code.abc_class as abc_class
import init


def main():
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = os.path.join('./', 'params.yml')
    path = yaml.load(open(input_path, 'r'))['path']

    logPath = path['output']
    logging.basicConfig(
        format="%(levelname)s: %(name)s:  %(message)s",
        handlers=[logging.FileHandler("{0}/{1}.log".format(logPath, 'ABC_log')), logging.StreamHandler()],
        level=logging.DEBUG)

    logging.info('platform {}'.format(sys.platform))
    logging.info('python {}.{}.{}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
    logging.info('numpy {}'.format(np.__version__))
    logging.info('64 bit {}\n'.format(sys.maxsize > 2 ** 32))

    ####################################################################################################################
    # Preprocess
    ####################################################################################################################
    params = init.CreateParams()
    init.LES_TEST_data(params.data, params.physical_case, params.compare_pdf)
    g.TEST_sp = data.DataSparse(g.TEST, params.abc['num_training_points'])
    g.TEST_Model = model.NonlinearModel(g.TEST_sp, params.model, params.abc['algorithm'], params.algorithm,
                                        params.C_limits, params.compare_pdf)
    if params.parallel['N_proc'] > 1:
        g.par_process = parallel.Parallel(params.parallel['progressbar'], params.parallel['N_proc'])

    ####################################################################################################################
    # ABC algorithm
    ####################################################################################################################
    abc = abc_class.ABC(params.abc, params.algorithm, params.model['N_params'], params.parallel['N_proc'],
                        params.C_limits)
    abc.main_loop()
    np.savez(os.path.join(g.path['output'], 'accepted.npz'), C=g.accepted, dist=g.dist)

    logging.info('Accepted parameters and distances saved in {}'.format(os.path.join(g.path['output'], 'accepted.npz')))


if __name__ == '__main__':
    main()
