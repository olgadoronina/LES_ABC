import params
import logging
import data
import global_var as g
import numpy as np
import plotting
import init
import postprocess


uniform = 1
calibration = 0
IMCMC = 0

filename_calibration_all = './plots/calibration_all.npz'
filename_calibration = './plots/calibration.npz'
filename_accepted = './plots/accepted.npz'

if calibration:
    filename = filename_calibration
else:
    filename = filename_accepted

logging.basicConfig(format='%(levelname)s: %(name)s: %(message)s', level=logging.DEBUG)

####################################################################################################################
# Initial data
####################################################################################################################
# initialize = init.Init()
params = init.CreateParams()
init.LES_TEST_data(params.data, params.physical_case, params.compare_pdf, params.abc['summary_statistics'])
g.TEST_sp = data.DataSparse(g.TEST, params.abc['num_training_points'])


########################
g.accepted = np.load(filename)['C']
g.dist = np.load(filename)['dist']
########################
#
# if IMCMC:
#     S_init = np.load(filename_calibration)['C']
#     S_dist = np.load(filename_calibration)['dist']
#     g.accepted = np.vstack((g.accepted, S_init))

if calibration:
    C_limits = params.C_limits
    num_bin_joint = 10
    N_each = 10
    dist = np.load(filename_calibration_all)['S_init'][:, -1]
    plotting.dist_pdf(dist, params.x, params.output['output_path'])

else:
    num_bin_joint = 20
    N_each = 100
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
params.algorithm['N_each'] = N_each
postproc = postprocess.PostprocessABC(C_limits, eps, num_bin_joint, params.output['output_path'])


if uniform:
    new_eps = 2000
    g.accepted = g.accepted[g.dist < new_eps]
    g.dist = g.dist[g.dist < new_eps]
    logging.info('accepted {} values ({}%)'.format(len(g.accepted),
                                                   round(len(g.accepted) / params.algorithm['N_total'] * 100, 2)))


postproc.calc_final_C()
postproc.plot_marginal_pdf()
if not calibration:
    #
    # postproc.plot_eps()
    # postproc.plot_scatter()
    # # postproc.scatter_animation()
    postproc.plot_compare_tau(scale='TEST_M', MCMC=0)
    postproc.plot_compare_tau(scale='TEST', MCMC=0)


















































































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



