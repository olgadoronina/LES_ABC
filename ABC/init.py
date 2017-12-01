import logging

import abc_class
import data
import filter
import global_var as g
import model
import numpy as np
import parallel
import params


class NPoints():
    def __init__(self):
        self.each = params.N_each
        self.params_in_task = params.N_params_in_task
        self.training = params.M
        if params.N_params_force:  # ignore the order and use this number of params
            self.params = params.N_params_force
        else:
            num_param = {'1': 1, '2': 4, '3': 6, '4': 9, '5': 10}
            self.params = num_param[str(params.ORDER)]


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

    def LES_TEST_data(self):

        if params.LOAD:  # Load filtered data from file
            loadfile_LES = params.data_folder + 'LES.npz'
            loadfile_TEST = params.data_folder + 'TEST.npz'

            logging.info("Load LES and TEST data")
            LES_data = np.load(loadfile_LES)
            TEST_data = np.load(loadfile_TEST)

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

            logging.info('Filter HIT data')
            LES_data = filter.filter3d(data=HIT_data, scale_k=params.LES_scale)
            TEST_data = filter.filter3d(data=HIT_data, scale_k=params.TEST_scale)
            # utils.spectral_density([LES_data['u'], LES_data['v'], LES_data['w']], 'LES')
            # utils.spectral_density([TEST_data['u'], TEST_data['v'], TEST_data['w']], 'TEST')
            logging.info('Writing files')
            np.savez(params.data_folder + 'LES.npz', **LES_data)
            np.savez(params.data_folder + 'TEST.npz', **TEST_data)

            # map_bounds = np.linspace(np.min(HIT_data['v'][:, :, 127]), np.max(HIT_data['v'][:, :, 127]), 10)
            # plot.imagesc([HIT_data['v'][:, :, 127], LES_data['v'][:, :, 127], TEST_data['v'][:, :, 127]],
            #              map_bounds, name='compare_velocity', titles=[r'$DNS$', r'$LES$', r'$TEST$'])
            # plt.show()

        dx = np.divide(params.lx, params.N_points)
        LES_delta = 1 / params.LES_scale
        TEST_delta = 1 / params.TEST_scale
        logging.info('Create LES class')
        g.LES = data.Data(LES_data, LES_delta, params.HOMOGENEOUS, dx)
        logging.info('Create TEST class')
        g.TEST = data.Data(TEST_data, TEST_delta, params.HOMOGENEOUS, dx)
        del LES_data, TEST_data

    def TEST_sparse_data(self):
        g.TEST_sp = data.DataSparse(g.TEST, params.M, )

    def model_on_sparse_TEST_data(self):
        g.TEST_Model = model.NonlinearModel(g.TEST_sp, self.N)

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
        postproc = abc_class.PostprocessABC(params.C_limits, self.eps, self.N)
        return postproc

    # def init_plotting(self):
    #
    #     import matplotlib.pyplot as plt
    #     import matplotlib as mpl
    #
    #     # mpl.style.use(['dark_background','mystyle'])
    #     # mpl.style.use(['mystyle'])
    #     plt.style.use('seaborn-white')
    #     plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
    #     plt.rc('text', usetex=True)
    #     plt.rcParams['mathtext.fontset'] = 'custom'
    #     plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    #     plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    #     plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    #     # plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    #     plt.rcParams['font.size'] = 18
    #     plt.rcParams['axes.labelsize'] = 18
    #     # plt.rcParams['axes.labelweight'] = 'bold'
    #     plt.rcParams['axes.titlesize'] = 20
    #     plt.rcParams['xtick.labelsize'] = 14
    #     plt.rcParams['ytick.labelsize'] = 14
    #     plt.rcParams['legend.fontsize'] = 16
    #     plt.rcParams['figure.titlesize'] = 22
