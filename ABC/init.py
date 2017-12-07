import logging

import matplotlib as mpl

mpl.use('pdf')
import matplotlib.pyplot as plt


import abc_class
import data
import filter
import utils
import global_var as g
import model
import numpy as np
import parallel
import params
import plotting


class NPoints():

    def __init__(self):
        self.bin_joint = params.num_bin_joint
        self.each = params.N_each
        self.training = params.M
        if params.N_params_force:  # ignore the order and use this number of params
            self.params = params.N_params_force
        else:
            num_param = {'1': 1, '2': 4, '3': 6, '4': 9, '5': 10}
            self.params = num_param[str(params.ORDER)]
        if self.params == 1:       # ignore the number and use 0
            self.params_in_task = 0
        else:
            self.params_in_task = params.N_params_in_task
        self.total = self.each ** self.params

class Init(object):

    def __init__(self):

        g.eps = params.eps
        g.bins = params.bins
        g.domain = params.domain

        self.C_limits = params.C_limits

        self.N_proc = params.N_proc
        self.N = NPoints()

        logging.info('Number of parameters = {}'.format(self.N.params))
        logging.info('Number of samples per interval = {}'.format(self.N.each))
        logging.info('Number of parameters per task = {}\n'.format(self.N.params_in_task))

    def plotting(self):

        # mpl.style.use(['dark_background','mystyle'])
        # mpl.style.use(['mystyle'])

        # mpl.rcParams['figure.figsize'] = 6.5, 2.2
        mpl.rcParams['font.size'] = 10
        mpl.rcParams['font.family'] = 'Times New Roman'
        mpl.rc('text', usetex=True)
        mpl.rcParams['axes.labelsize'] = plt.rcParams['font.size']
        mpl.rcParams['axes.titlesize'] = 1.5 * plt.rcParams['font.size']
        mpl.rcParams['legend.fontsize'] = plt.rcParams['font.size']
        mpl.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
        mpl.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
        # plt.rcParams['savefig.dpi'] = 2 * plt.rcParams['savefig.dpi']
        mpl.rcParams['xtick.major.size'] = 3
        mpl.rcParams['xtick.minor.size'] = 3
        mpl.rcParams['xtick.major.width'] = 1
        mpl.rcParams['xtick.minor.width'] = 1
        mpl.rcParams['ytick.major.size'] = 3
        mpl.rcParams['ytick.minor.size'] = 3
        mpl.rcParams['ytick.major.width'] = 1
        mpl.rcParams['ytick.minor.width'] = 1
        mpl.rcParams['legend.frameon'] = False
        # plt.rcParams['legend.loc'] = 'center left'
        plt.rcParams['axes.linewidth'] = 1

        g.plot = plotting.Plot(params.plot_folder, params.PLOT_INIT_INFO)


    def LES_TEST_data(self):

        dx = np.divide(params.lx, params.N_points)

        if params.LOAD:  # Load filtered data from file
            loadfile_LES = params.data_folder + 'LES.npz'
            loadfile_TEST = params.data_folder + 'TEST.npz'

            logging.info("Load LES and TEST data")
            LES_data = np.load(loadfile_LES)
            TEST_data = np.load(loadfile_TEST)

            g.plot.map_bounds = np.linspace(np.min(LES_data['v'][:, :, 127]), np.max(LES_data['v'][:, :, 127]), 9)

        else:  # Filter HIT data
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

            utils.spectral_density([HIT_data['u'], HIT_data['v'], HIT_data['w']], dx, params.N_points,
                                   params.plot_folder + 'DNS')

            logging.info('Filter HIT data')
            LES_data = filter.filter3d(data=HIT_data, scale_k=params.LES_scale, dx=dx, N_points=params.N_points)
            TEST_data = filter.filter3d(data=HIT_data, scale_k=params.TEST_scale, dx=dx, N_points=params.N_points)

            logging.info('Writing files')
            np.savez(params.data_folder + 'LES.npz', **LES_data)
            np.savez(params.data_folder + 'TEST.npz', **TEST_data)

            g.plot.map_bounds = np.linspace(np.min(LES_data['v'][:, :, 127]), np.max(LES_data['v'][:, :, 127]), 9)

            if params.PLOT_INIT_INFO:
                g.plot.compare_filter_fields(HIT_data, LES_data, TEST_data)
            del HIT_data

        LES_delta = 1 / params.LES_scale
        TEST_delta = 1 / params.TEST_scale
        logging.info('Create LES class')
        g.LES = data.Data(LES_data, LES_delta, params.HOMOGENEOUS, dx, params.PLOT_INIT_INFO)
        logging.info('Create TEST class')
        g.TEST = data.Data(TEST_data, TEST_delta, params.HOMOGENEOUS, dx, params.PLOT_INIT_INFO)

        if params.PLOT_INIT_INFO:
            logging.info('Calculate spectras')
            utils.spectral_density([LES_data['u'], LES_data['v'], LES_data['w']], dx, params.N_points,
                                   params.plot_folder+'LES')
            utils.spectral_density([TEST_data['u'], TEST_data['v'], TEST_data['w']], dx, params.N_points,
                                   params.plot_folder + 'test')

            logging.info('Plot initial data info')
            g.plot.vel_fields(scale='LES')
            g.plot.vel_fields(scale='TEST')
            g.plot.sigma_field(scale='LES')
            g.plot.sigma_field(scale='TEST')
            g.plot.sigma_pdf()

        del LES_data, TEST_data

    def TEST_sparse_data(self):
        g.TEST_sp = data.DataSparse(g.TEST, params.M)

    def model_on_sparse_TEST_data(self):
        g.TEST_Model = model.NonlinearModel(g.TEST_sp, self.N, self.C_limits)

    def parallel(self):
        if self.N_proc > 1:
            g.par_process = parallel.Parallel(processes=self.N_proc, N_total=self.N.each ** self.N.params,
                                              progressbar=params.PROGRESSBAR)

    def ABC_algorithm(self):

        abc_init = abc_class.ABC(N=self.N, C_limits=self.C_limits, eps=params.eps)
        return abc_init


class InitPostProcess(object):

    def __init__(self, eps):
        self.eps = eps
        self.N = NPoints()

    def postprocessing(self):
        postproc = abc_class.PostprocessABC(params.C_limits, self.eps, self.N, params.plot_folder, params.num_bin_joint)
        return postproc


