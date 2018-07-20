from __future__ import division
import logging
import sys
import os
import yaml
import numpy as np

from mpi4py import MPI


import abc_code.global_var as g
import abc_code.data as data
import abc_code.model as model
import abc_code.parallel as parallel
import abc_code.abc_class as abc_class
import init


def main():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()


    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = os.path.join('./', 'params.yml')
    path = yaml.load(open(input_path, 'r'))['path']

    logPath = path['output']

    logging.basicConfig(
        format="%(levelname)s: %(name)s:  %(message)s",
        handlers=[logging.FileHandler("{0}/{1}.log".format(logPath, 'ABC_log_{}'.format(rank))), logging.StreamHandler()],
        level=logging.DEBUG)



    logging.debug('Name = {}, Rank {} of Size {}'.format(name, rank, size))
    if rank == 0:
        logging.info('platform {}'.format(sys.platform))
        logging.info('python {}.{}.{}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
        logging.info('numpy {}'.format(np.__version__))
        logging.info('64 bit {}\n'.format(sys.maxsize > 2 ** 32))

        ####################################################################################################################
        # Preprocess
        ####################################################################################################################

    params = init.CreateParams()
    if params.data['load']:
        g.TEST_sp = data.DataSparse(os.path.join(params.data['data_path']), 1)
    else:
        init.LES_TEST_data(params.data, params.physical_case, params.compare_pdf)
        g.TEST_sp = data.DataSparse(os.path.join(params.data['data_path']), 0, g.TEST, params.abc['num_training_points'])
        g.LES = None
        g.TEST = None

    g.TEST_Model = model.NonlinearModel(g.TEST_sp, params.model, params.abc['algorithm'], params.algorithm,
                                        params.C_limits, params.compare_pdf)
    logging.debug('here')

        # comm.bcast(params, root=0)
        # logging.debug('done')
        # comm.bcast(g.TEST_Model, root=0)
        # logging.debug('done')
        # comm.bcast(g.TEST_sp, root=0)
        # logging.debug('done')

    comm.Barrier()
    logging.info('Model {}'.format(g.TEST_Model))
    logging.info('TEST_sp {}'.format(g.TEST_sp))
    if params.parallel['N_proc'] > 1:
        g.par_process = parallel.Parallel(params.parallel['progressbar'], params.parallel['N_proc'])

    ####################################################################################################################
    # ABC algorithm
    ####################################################################################################################
    abc = abc_class.ABC(params.abc, params.algorithm, params.model['N_params'], params.parallel['N_proc'],
                        params.C_limits)
    abc.main_loop()
    comm.Barrier()
    np.savez(os.path.join(g.path['output'], 'accepted_{}.npz'.format(rank)), C=g.accepted, dist=g.dist)

    logging.info('Accepted parameters and distances saved in {}'.format(os.path.join(g.path['output'],
                                                                                     'accepted_{}.npz'.format(rank))))


if __name__ == '__main__':
    main()
