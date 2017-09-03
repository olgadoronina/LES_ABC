import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from math import *
import glob, os
from importlib import reload
# mpl.use('PDF')
mpl.rcParams['text.usetex']=True
mpl.rcParams['text.latex.unicode']=True
mpl.rcParams['backend']='MacOSX'
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
# mpl.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica Neue']})
# mpl.rc('font', **{'family': 'serif', 'serif': ['Times']})
mpl.rc('axes.formatter', use_mathtext=False, )
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['cm']
mpl.rc('axes', labelpad=3.0, )

plt = reload(plt)


###################################################################
# os.chdir("./HIT_LES/test_N64/analysis/")
# # ####################################################################################################################
# My spectra
# ######################################################################################################################
os.chdir("./ABC/plots/")
files = glob.glob("*.spectra")

for k in range(len(files)):
    file = files[k]
    f = open(file, 'r')
    label = file[:3]
    data = np.array(f.readlines()).astype(np.float)
    x = np.arange(len(data))
    plt.loglog(x, data, '-', linewidth=3, label=label)
    y = 7.2e14*np.power(x, -5/3)
    plt.loglog(x, y, 'r--')
plt.title('Spectrum', fontsize=20)
plt.ylabel(r'$E$', fontsize=20)
plt.xlabel(r'k', fontsize=20)
plt.axis(ymin=1e6)
plt.legend(loc=0)
plt.show()
# ########################################################################################################################
# # U spectra
# ######################################################################################################################
# files = glob.glob("test_LES_N64-*_u.spectra")
# files.sort()
# num_of_files = len(files)
# print(num_of_files)
# for k in range(0, num_of_files, 20):
#     file = files[k]
#     tstep = file[13:16]
#     if file[16] != '_':
#         tstep = tstep + file[16]
#     f = open(file, 'r')
#     title = f.readline()
#     y_label = f.readline()
#     sth = f.readline()
#     data = np.array(f.readlines()).astype(np.float)
#     plt.loglog(data, '-o', label='step '+tstep)
# plt.title('velocity PSD', fontsize=24)
# plt.ylabel(r'$\widehat{u_i}^*\widehat{u_i}$', fontsize=24)
# plt.axis(ymin=1e5, ymax=1.2e11)
# plt.legend(loc=0)
# plt.show()
#
# ########################################################################################################################
# # Omega spectra
# ######################################################################################################################
# files = glob.glob("test_LES_N64-*_omega.spectra")
# files.sort()
# num_of_files = len(files)
# print(num_of_files)
# for k in range(0, num_of_files, 20):
#     file = files[k]
#     tstep = file[13:16]
#     if file[16] != '_':
#         tstep = tstep + file[16]
#     f = open(file, 'r')
#     title = f.readline()
#     y_label = f.readline()
#     sth = f.readline()
#     data = np.array(f.readlines()).astype(np.float)
#     plt.loglog(data, '-o', label='step '+tstep)
# plt.title('vorticity PSD', fontsize=24)
# plt.ylabel(r'$\widehat{\omega_i}^*\widehat{\omega_i}$', fontsize=24)
# plt.axis(ymin=1e7, ymax=1e12)
# plt.legend(loc=0)
# plt.show()
#
# ########################################################################################################################
# # Enstrophy histogram
# ########################################################################################################################
# files = glob.glob("test_LES_N64-*_enst.hist")
# num_of_files = len(files)
# print(num_of_files)
# for k in range(0, num_of_files, 10):
#     file = files[k]
#     tstep = file[13:16]
#     if file[16] != '_':
#         tstep = tstep + file[16]
#     f = open(file, 'r')
#     title = f.readline()
#     y_label = f.readline()
#     mean = float(f.readline())
#     variance = float(f.readline())
#     minimum = float(f.readline())
#     maximum = float(f.readline())
#     bin_width = float(f.readline())
#     number_of_bins = int(f.readline())
#     x = np.linspace(minimum, maximum, num=number_of_bins)+bin_width/2
#     data = np.array(f.readlines()).astype(np.float)
#     plt.plot(x, data, '-o', label='step ' + tstep + ', max={}'.format(maximum))
# plt.xlabel(r'$\Omega$', fontsize=24)
# plt.ylabel(r'$\mathrm{pdf}$', fontsize=24)
# plt.axis(xmin=0, xmax=30)
# plt.legend(loc=0)
# plt.show()
