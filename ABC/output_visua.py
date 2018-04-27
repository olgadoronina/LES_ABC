import params
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import logging
import sys

import global_var as g
import init
import numpy as np
import plotting


calibration = 1
IMCMC = 0

if calibration:
    filename_all = './plots/calibration_all.npz'
    filename = './plots/calibration.npz'
else:
    if IMCMC:
        filename_calibration = './plots/calibration.npz'
    filename = './plots/accepted.npz'

logging.basicConfig(format='%(levelname)s: %(name)s: %(message)s', level=logging.DEBUG)

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
abc = initialize.ABC_algorithm()
del initialize

# ########################
g.accepted = np.load(filename)['C']
g.dist = np.load(filename)['dist']
if IMCMC:
    S_init = np.load(filename_calibration)['C']
    S_dist = np.load(filename_calibration)['dist']
    # S_init[:, 0] = np.sqrt(-S_init[:, 0] / 2)
    g.accepted = np.vstack((g.accepted, S_init))

if calibration:
    # g.accepted[:, 0] = np.sqrt(-g.accepted[:, 0] / 2)
    C_limits = params.C_limits

    num_bin_joint = 10
    dist = np.load(filename_all)['S_init'][:, -1]
    plotting.dist_pdf(dist)
else:
    num_bin_joint = 10
    C_limits = params.C_limits
    # C_limits = np.zeros((10, 2))
    # C_limits[0] = [np.min(g.accepted[:, 0]), np.max(g.accepted[:, 0])]
    # C_limits[1] = [np.min(g.accepted[:, 1]), np.max(g.accepted[:, 1])]
    # C_limits[2] = [np.min(g.accepted[:, 2]), np.max(g.accepted[:, 2])]
    # C_limits[3] = [np.min(g.accepted[:, 3]), np.max(g.accepted[:, 3])]
    # C_limits[4] = [np.min(g.accepted[:, 4]), np.max(g.accepted[:, 4])]
    # C_limits[5] = [np.min(g.accepted[:, 5]), np.max(g.accepted[:, 5])]
# # # #########################
eps = g.eps
# eps = new_eps
print(g.C_limits)
initialize = init.InitPostProcess(eps, g.C_limits, num_bin_joint)
postproc = initialize.postprocessing()


postproc.calc_final_C()
postproc.plot_marginal_pdf()

if not calibration:
    # # postproc.plot_eps()
    postproc.plot_scatter()
    # # postproc.scatter_animation()
    postproc.plot_compare_tau(scale='TEST_M', MCMC=0)
    postproc.plot_compare_tau(scale='TEST', MCMC=0)
    # postproc.plot_compare_tau('LES')

# new_eps = 25
# g.accepted = g.accepted[g.dist < new_eps]
# g.dist = g.dist[g.dist < new_eps]
# logging.info('accepted {} values ({}%)'.format(len(g.accepted), round(len(g.accepted) / abc.N.total * 100, 2)))

















    # if g.plot.plot_info:
    # #     logging.info('Plot initial data info')
    # #     # g.plot.vel_fields(scale='LES')
    # #     # g.plot.vel_fields(scale='TEST')
    # #     # g.plot.sigma_field(scale='LES')
    # #     # g.plot.sigma_field(scale='TEST')
    # #     # g.plot.sigma_pdf()
    #     g.plot.S_pdf()
    #     g.plot.A_compare()
        # g.plot.spectra()

    ####################################################################################################################
    # logging.info('Strain tensors')
    # g.HIT.strain_tensor()
    # g.LES.strain_tensor()
    # g.TEST.strain_tensor()
    # g.LES_sp.A = utils.sparse_dict(g.LES.A, M)
    # g.TEST_sp.A = utils.sparse_dict(g.TEST.A, M)
    # plot.A_compare(g.TEST_sp.A, axarr, titles, M=M, color=colors[k])
    ####################################################################################################################












# mpl.style.use(['dark_background','mystyle'])
# # mpl.style.use(['mystyle'])
#
# # mpl.rcParams['figure.figsize'] = 6.5, 2.2
# plt.rcParams['figure.autolayout'] = True
#
# mpl.rcParams['font.size'] = 10
# mpl.rcParams['font.family'] = 'Times New Roman'
# mpl.rc('text', usetex=True)
# mpl.rcParams['axes.labelsize'] = plt.rcParams['font.size']
# mpl.rcParams['axes.titlesize'] = 1.5 * plt.rcParams['font.size']
# mpl.rcParams['legend.fontsize'] = plt.rcParams['font.size']
# mpl.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
# mpl.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
# # plt.rcParams['savefig.dpi'] = 2 * plt.rcParams['savefig.dpi']
# mpl.rcParams['xtick.major.size'] = 3
# mpl.rcParams['xtick.minor.size'] = 3
# mpl.rcParams['xtick.major.width'] = 1
# mpl.rcParams['xtick.minor.width'] = 1
# mpl.rcParams['ytick.major.size'] = 3
# mpl.rcParams['ytick.minor.size'] = 3
# mpl.rcParams['ytick.major.width'] = 1
# mpl.rcParams['ytick.minor.width'] = 1
# # mpl.rcParams['legend.frameon'] = False
# # plt.rcParams['legend.loc'] = 'center left'
# plt.rcParams['axes.linewidth'] = 1

########################################################################################################################
# Initial data
########################################################################################################################
# fig = pickle.load(open(params.plot_folder + 'LES_velocities', 'rb'))
# plt.show()
#
# fig = pickle.load(open(params.plot_folder + 'TEST_velocities', 'rb'))
# plt.show()
#
# fig = pickle.load(open(params.plot_folder+'sigma_TEST', 'rb'))
# fig.set_title[r'$\sigma_{11}$', r'$\sigma_{12}$', r'$\sigma_{13}$']
# plt.show()
#
# fig = pickle.load(open(params.plot_folder+'TEST', 'rb'))
#
# plt.show()




























































# from lxml import etree
# yhttp://lxml.de/tutorial.html
# http://www.diveintopython3.net/xml.html




# class VTK_XML_Serial_Structured:
#     def __init__(self, extent):
#         extent_str = '{} {} {} {} {} {}'.format(extent[0], extent[1]-1, extent[2], extent[3]-1, extent[4], extent[5]-1)
#         self.root = etree.Element('VTKFile', type='StructuredGrid', version='0.1', byte_order="LittleEndian")
#         grid_type = etree.SubElement(self.root, 'StructuredGrid', WholeExtent=extent_str)
#         piece = etree.SubElement(grid_type, 'Piece', Extent=extent_str)
#         self.pointsdata = etree.SubElement(piece, 'PointData')
#         self.celldata = etree.SubElement(piece, 'CellData')
#         self.coord = etree.SubElement(piece, 'Points')
#
#     def __str__(self):
#         logging.info(etree.tostring(self.root, pretty_print=True, xml_declaration=True))
#
#
#     def coords_to_string(self, X):
#         string = str()
#         a, Nx, Ny, Nz = X.shape
#         for i in range(Nx):
#             for j in range(Ny):
#                 for k in range(Nz):
#                     string += '{} {} {}\n'.format(repr(X[0, i, j, k]), repr(X[1, i, j, k]), repr(X[2, i, j, k]))
#         return string
#
#     def data_to_string(self, a):
#         string = str()
#         Nx, Ny, Nz = a.shape
#         for i in range(Nx):
#             for j in range(Ny):
#                 for k in range(Nz):
#                     string += '{}\n'.format(repr(a[i, j, k]))
#         return string
#
#     def add_pointdata(self, data, name):
#         dataarray = etree.SubElement(self.pointsdata, 'DataArray', Scalar=name, type='Float32', format="ascii", NumberOfComponents="1")
#         dataarray.text = self.data_to_string(data)
#
#     def add_points(self, coord):
#         coords = etree.SubElement(self.coord, 'DataArray', type='Float32', format="ascii", NumberOfComponents="3")
#         coords.text = self.coords_to_string(coord)
#
#
#     def write_to_file(self, filename, X, U):
#         # self.add_points(X)
#         # self.add_pointdata(U[0], 'u')
#         # self.add_pointdata(U[1], 'v')
#         # self.add_pointdata(U[2], 'w')
#         doc = etree.ElementTree(self.root)
#         doc.write(filename, pretty_print=True, xml_declaration=True)



