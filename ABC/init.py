import logging
import abc_class
import data
import utils
import global_var as g
import model
import numpy as np
import parallel
import params
import postprocess


class NPoints():

    def __init__(self):

        self.bin_joint = params.num_bin_joint

        self.training = params.M
        self.proc = params.N_proc

        # define number of params
        if params.N_params_force:  # ignore the order and use this number of params
            self.params = params.N_params_force
        else:
            num_param = {'1': 1, '2': 4, '3': 6, '4': 9, '5': 10}
            self.params = num_param[str(params.ORDER)]

        # define number samples per task
        if self.params == 1 or params.MCMC == 1 or params.sampling != 'uniform':       # ignore the number and use 0
            self.params_in_task = 0
        else:
            self.params_in_task = params.N_params_in_task

        # define number of samples
        if params.MCMC == 1:
            self.total = params.N_total
            self.chain = int(self.total/self.proc)
        elif params.MCMC == 2:
            self.total = params.N_total
            self.calibration = params.N_calibration
            self.each = int(round(self.calibration**(1/self.params)))
            self.chain = int(self.total/self.proc)
        else:
            self.each = params.N_each
            self.calibration = self.each ** self.params
            self.total = self.each ** self.params


class Init(object):

    def __init__(self):

        g.eps = params.eps
        g.bins = params.bins
        g.domain = params.domain

        self.N = NPoints()
        g.N = self.N
        self.C_limits = params.C_limits[:self.N.params].copy()
        g.C_limits = self.C_limits[:self.N.params].copy()

        # g.C_limits[0, 0] = - 2 * params.C_limits[0, 1] ** 2
        # g.C_limits[0, 1] = - 2 * params.C_limits[0, 0] ** 2

        self.LES_scale = params.LES_scale
        self.TEST_scale = params.TEST_scale

        logging.info('Number of parameters = {}'.format(self.N.params))
        logging.info('Number of processors = {}'.format(self.N.proc))
        if params.MCMC == 1:
            g.std = np.sqrt(params.var[:self.N.params])
            logging.info('Number of accepted samples per MCMC chain = {}\n'.format(self.N.chain))
        elif params.MCMC == 2:
            if self.N.each > 0:
                logging.info('Number samples on calibration step = {}'.format(self.N.calibration))
                logging.info('Number of samples in each dimension on calibration step = {}'.format(self.N.each))
                logging.info('Number of parameters per task on calibration step = {}\n'.format(self.N.params_in_task))
            else:
                logging.info('Download calibration data')
            logging.info('Number of accepted samples per MCMC chain = {}\n'.format(self.N.chain))
        else:
            logging.info('Number of samples per interval = {}'.format(self.N.each))
            logging.info('Number of parameters per task = {}\n'.format(self.N.params_in_task))

    def LES_TEST_data(self):

        def load_HIT_data():
            datafile = dict()
            if params.DATA == 'JHU_data':
                datafile['u'] = params.data_folder + 'HIT_u.bin'
                datafile['v'] = params.data_folder + 'HIT_v.bin'
                datafile['w'] = params.data_folder + 'HIT_w.bin'
                type_of_bin_data = np.float32
            elif params.DATA == 'CU_data':
                datafile['u'] = params.data_folder + 'Velocity1_003.rst'
                datafile['v'] = params.data_folder + 'Velocity2_003.rst'
                datafile['w'] = params.data_folder + 'Velocity3_003.rst'
                type_of_bin_data = np.float64

            logging.info('Load HIT data')
            HIT_data = dict()
            data_shape = (params.N_point, params.N_point, params.N_point)
            for i in ['u', 'v', 'w']:
                HIT_data[i] = np.reshape(np.fromfile(datafile[i], dtype=type_of_bin_data), data_shape)
            for key, value in HIT_data.items():
                HIT_data[key] = np.swapaxes(value, 0, 2)  # to put x index in first place
            for i in ['u', 'v', 'w']:
                for j in ['u', 'v', 'w']:
                    HIT_data[i + j] = np.multiply(HIT_data[i], HIT_data[j])

            return HIT_data

        dx = np.divide(params.lx, params.N_points)

        if params.LOAD:  # Load filtered data from file
            logging.info("Load LES data")
            loadfile_LES = params.data_folder + 'LES.npz'
            LES_data = np.load(loadfile_LES)

            if self.TEST_scale:
                logging.info("Load TEST data")
                loadfile_TEST = params.data_folder + 'TEST.npz'
                TEST_data = np.load(loadfile_TEST)
            else:
                HIT_data = load_HIT_data()

        else:  # Filter HIT data
            HIT_data = load_HIT_data()

            logging.info('Filter HIT data')
            LES_data = utils.filter3d(data=HIT_data, scale_k=params.LES_scale, dx=dx, N_points=params.N_points)
            TEST_data = None
            logging.info('Writing file')
            np.savez(params.data_folder + 'LES.npz', **LES_data)
            if self.TEST_scale:
                TEST_data = utils.filter3d(data=HIT_data, scale_k=params.TEST_scale, dx=dx, N_points=params.N_points)
                logging.info('Writing file')
                np.savez(params.data_folder + 'TEST.npz', **TEST_data)

        if self.TEST_scale:
            LES_delta = 1 / params.LES_scale
            TEST_delta = 1 / params.TEST_scale
            logging.info('Create LES class')
            g.LES = data.Data(LES_data, LES_delta, params.HOMOGENEOUS, dx, params.PLOT_INIT_INFO)
            logging.info('Create TEST class')
            g.TEST = data.Data(TEST_data, TEST_delta, params.HOMOGENEOUS, dx, params.PLOT_INIT_INFO)
        else:
            DNS_delta = params.lx[0]
            LES_delta = 1 / params.LES_scale
            logging.info('Create LES class')
            g.LES = data.Data(HIT_data, DNS_delta, params.HOMOGENEOUS, dx, params.PLOT_INIT_INFO)
            logging.info('Create TEST class')
            g.TEST = data.Data(LES_data, LES_delta, params.HOMOGENEOUS, dx, params.PLOT_INIT_INFO)

        if self.TEST_scale:
            del TEST_data, LES_data
        else:
            del HIT_data, LES_data

    def TEST_sparse_data(self):
        g.TEST_sp = data.DataSparse(g.TEST, params.M)

    def model_on_sparse_TEST_data(self):
        g.TEST_Model = model.NonlinearModel(g.TEST_sp, params.HOMOGENEOUS, self.N, self.C_limits, params.MCMC)

    def parallel(self):
        if self.N.proc > 1:
            g.par_process = parallel.Parallel(self.N.total, params.PROGRESSBAR, self.N.proc)

    def ABC_algorithm(self):

        if params.sampling == 'random':
            sampling_func = utils.sampling_random
        elif params.sampling == 'uniform':
            sampling_func = utils.sampling_uniform_grid
        elif params.sampling == 'sobol':
            sampling_func = utils.sampling_sobol
        elif params.sampling == 'MCMC':
            sampling_func = utils.sampling_initial_for_MCMC
        else:
            logging.error('Incorrect sampling type {}'.format(params.sampling))
            exit()

        logging.info('Sampling is {}'.format(params.sampling))

        abc_init = abc_class.ABC(N=self.N, form_C_array=sampling_func)
        return abc_init



