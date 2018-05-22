import logging
import data
import utils
import global_var as g
import numpy as np
import os
import params


class CreateParams:
    """Define all input parameters using params.py file and store it in object"""

    def __init__(self):

        self.physical_case = params.physical_case
        self.data = params.data
        self.output = params.output
        self.parallel = params.parallel
        self.abc = params.abc

        self.compare_pdf = self.define_compare_pdf_params()
        self.model = self.define_model_params(params.model)
        self.C_limits = np.array(params.C_limits[:self.model['N_params']])

        self.algorithm = self.define_algorithm_params(self.abc)

        logging.debug(
            'INPUT PARAMETERS:\n'
            '############################################################################\n'
            '{0}{1}{2}{3}{4}{5}{6}{7}{8}'
            '############################################################################'.format(
                'Physical case parameters:\n{}\n'.format(self.physical_case),
                'Data parameters:\n{}\n'.format(self.data),
                'Output path: \n{}\n'.format(self.output),
                'Parrallel regime parameters:\n{}\n'.format(self.parallel),
                'Parameters of pdf comparison:\n{}\n'.format(self.compare_pdf),
                'Limits for coefficients:\n{}\n'.format(self.C_limits),
                'Viscous model:\n{}\n'.format(self.model),
                'ABC parameters:\n{}\n'.format(self.abc),
                'Algorithm:\n{}\n'.format(self.algorithm)))

        self.print_params_summary()

    def define_compare_pdf_params(self):
        compare_pdf = params.compare_pdf
        g.pdf_params = compare_pdf
        edges = np.linspace(compare_pdf['domain'][0], compare_pdf['domain'][1], compare_pdf['bins'] + 1)
        x = (edges[1:] + edges[:-1]) / 2
        np.savetxt('./output/sum_stat_bins', x)
        return compare_pdf

    def define_model_params(self, params):
        model = dict()
        if params['N_params_force']:  # ignore the order and use this number of params
            model['N_params'] = params['N_params_force']
        else:
            num_param = {1: 1, 2: 4, 3: 6, 4: 9, 5: 10}
            model['N_params'] = num_param[params['order']]
        model['homogeneous'] = self.physical_case['homogeneous']
        return model

    def define_algorithm_params(self, abc):

        algorithm = dict()
        if abc['algorithm'] == 'acc-rej':
            algorithm = params.algorithm['acc-rej']
            assert algorithm['sampling'] in ['uniform', 'random', 'sobol'], \
                logging.error('Incorrect sampling type {}'.format(algorithm['sampling']))
            if self.model['N_params'] == 1 or algorithm['sampling'] != 'uniform':
                algorithm['N_params_in_task'] = 0
            algorithm['N_total'] = algorithm['N_each'] ** (self.model['N_params'])
            g.eps = algorithm['eps']
        elif abc['algorithm'] == 'MCMC':
            algorithm = params.algorithm['MCMC']
            algorithm['std'] = np.sqrt(self.C_limits[1]-self.C_limits[0]) * algorithm['var_multiplier']
            algorithm = self.define_chain_params(algorithm, self.parallel['N_proc'])
            g.eps = algorithm['eps']
        elif abc['algorithm'] == 'IMCMC':
            algorithm = params.algorithm['IMCMC']
            assert algorithm['sampling'] in ['uniform', 'random', 'sobol'], \
                logging.error('Incorrect sampling type {}'.format(algorithm['sampling']))
            algorithm = self.define_chain_params(algorithm, self.parallel['N_proc'])
            if self.model['N_params'] == 1 or algorithm['sampling'] != 'uniform':
                algorithm['N_params_in_task'] = 0
            if algorithm['N_params_in_task'] == 1:
                logging.warning('Does not work for IMCMC and N_params_in_task = 1, set N_params_in_task = 2')
                algorithm['N_params_in_task'] = 2
            algorithm['N_total'] = algorithm['N_each'] ** (self.model['N_params'] - algorithm['N_params_in_task'])
        elif abc['algorithm'] == 'AGM_MH':
            algorithm = params.algorithm['AGM_MH']
            algorithm['std'] = np.sqrt(self.C_limits[1] - self.C_limits[0]) * algorithm['var_multiplier']
        else:
            logging.error('Unknown algorithm = {}'.format(abc['algorithm']))
            exit()
        return algorithm

    def define_chain_params(self, algorithm, N_proc):
        if N_proc:
            algorithm['N_chain'] = int(algorithm['N_total_chain'] / N_proc)
            algorithm['N_total_chain'] = algorithm['N_chain'] * N_proc
        else:
            algorithm['N_chain'] = algorithm['N_total_chain']
        return algorithm

    def print_params_summary(self):

        if 'sampling' in self.algorithm.keys():
            logging.info('Number of samples per interval = {}'.format(self.algorithm['N_each']))
            logging.info('Number of parameters per task = {}'.format(self.algorithm['N_params_in_task']))
        if 'chains' in self.algorithm.keys():
            logging.info('Number of accepted samples per MCMC chain = {}'.format(self.algorithm['chains']['N_chain']))
        if 'calibration' in self.algorithm.keys():
            if self.algorithm['calibration']['N_total'] == 0:
                logging.info('Download calibration data')
            else:
                logging.info('Number samples on calibration step = {}'.format(self.algorithm['calibration']['N_total']))
                logging.info('Number of samples in each dimension on calibration step = {}'.format(
                    self.algorithm['calibration']['N_each']))
                logging.info('Number of parameters per task on calibration step = {}'.format(
                    self.algorithm['calibration']['N_params_in_task']))
        logging.info('Number of parameters = {}'.format(self.model['N_params']))
        logging.info('Number of processors = {}\n'.format(self.parallel['N_proc']))


def load_HIT_data(data_params, case_params):
    datafile = dict()
    if data_params['data_name'] == 'JHU_data':
        datafile['u'] = os.path.join(data_params['data_path'], 'HIT_u.bin')
        datafile['v'] = os.path.join(data_params['data_path'], 'HIT_v.bin')
        datafile['w'] = os.path.join(data_params['data_path'], 'HIT_w.bin')
        type_of_bin_data = np.float32
    elif data_params['data_name'] == 'CU_data':
        datafile['u'] = os.path.join(data_params['data_path'], 'Velocity1_003.rst')
        datafile['v'] = os.path.join(data_params['data_path'], 'Velocity2_003.rst')
        datafile['w'] = os.path.join(data_params['data_path'], 'Velocity3_003.rst')
        type_of_bin_data = np.float64
    logging.info('Load HIT data')
    HIT_data = dict()
    data_shape = (case_params['N_point'], case_params['N_point'], case_params['N_point'])
    for i in ['u', 'v', 'w']:
        HIT_data[i] = np.reshape(np.fromfile(datafile[i], dtype=type_of_bin_data), data_shape)
    for key, value in HIT_data.items():
        HIT_data[key] = np.swapaxes(value, 0, 2)  # to put x index in first place
    for i in ['u', 'v', 'w']:
        for j in ['u', 'v', 'w']:
            HIT_data[i + j] = np.multiply(HIT_data[i], HIT_data[j])
    return HIT_data


def LES_TEST_data(data_params, case_params, pdf_params):

    dx = np.array([case_params['lx']/case_params['N_point']]*3)

    if data_params['load']:  # Load filtered data from file
        logging.info("Load LES data")
        loadfile_LES = os.path.join(data_params['data_path'], 'LES.npz')
        LES_data = np.load(loadfile_LES)

        if case_params['TEST_scale']:
            logging.info("Load TEST data")
            loadfile_TEST = os.path.join(data_params['data_path'], 'TEST.npz')
            TEST_data = np.load(loadfile_TEST)
        else:
            HIT_data = load_HIT_data(data_params, case_params)

    else:  # Filter HIT data
        HIT_data = load_HIT_data(data_params, case_params)

        logging.info('Filter HIT data')
        LES_data = utils.filter3d(data=HIT_data, scale_k=case_params['LES_scale'],
                                  dx=dx, N_points=case_params['N_points'])
        TEST_data = None
        logging.info('Writing file')
        np.savez(os.path.join(data_params['data_path'], 'LES.npz'), **LES_data)
        if case_params['TEST_scale']:
            TEST_data = utils.filter3d(data=HIT_data, scale_k=case_params['TEST_scale'],
                                       dx=dx, N_points=case_params['N_points'])
            logging.info('Writing file')
            np.savez(os.path.join(data_params['data_path'], 'TEST.npz'), **TEST_data)

    if case_params['TEST_scale']:
        LES_delta = 1 / case_params['LES_scale']
        TEST_delta = 1 / case_params['TEST_scale']
        logging.info('Create LES class')
        g.LES = data.Data(LES_data, LES_delta, dx, pdf_params)
        logging.info('Create TEST class')
        g.TEST = data.Data(TEST_data, TEST_delta, dx, pdf_params)
        del TEST_data, LES_data
    else:
        DNS_delta = case_params['lx']/case_params['N_point']
        LES_delta = 1 / case_params['LES_scale']
        logging.info('Create LES class')
        g.LES = data.Data(HIT_data, DNS_delta, dx, pdf_params)
        logging.info('Create TEST class')
        g.TEST = data.Data(LES_data, LES_delta, dx, pdf_params)
        del HIT_data, LES_data





