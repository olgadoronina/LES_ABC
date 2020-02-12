import yaml
import os
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
from matplotlib import cm
import numpy as np
import string

# plt.style.use('dark_background')

# # thesis
# single_column = 235
# oneandhalf_column = 352
# double_column = 470

# paper twocolumn elsevair
single_column = 252
oneandhalf_column = 397
double_column = 522
text_height = 682/ 72.27

def fig_size(width_column):
    fig_width_pt = width_column
    inches_per_pt = 1.0 / 72.27  # Convert pt to inches
    golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = fig_width * golden_mean  # height in inches
    return fig_width, fig_height

mpl.rcParams['font.size'] = 9
mpl.rcParams['axes.titlesize'] = 1.2 * plt.rcParams['font.size']
mpl.rcParams['axes.labelsize'] = plt.rcParams['font.size']
mpl.rcParams['legend.fontsize'] = plt.rcParams['font.size']
mpl.rcParams['xtick.labelsize'] = 0.8*plt.rcParams['font.size']
mpl.rcParams['ytick.labelsize'] = 0.8*plt.rcParams['font.size']
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rc('text', usetex=True)
# plt.rcParams['savefig.dpi'] = 2 * plt.rcParams['savefig.dpi']
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.width'] = 0.5
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.width'] = 1
# mpl.rcParams['legend.frameon'] = False
# plt.rcParams['legend.loc'] = 'center left'
plt.rcParams['axes.linewidth'] = 1

#######################################################################################################################
#
#######################################################################################################################
params_names = [r'$C_1$', r'$C_2$', r'$C_3$']
N_params = 3
path_base = '../ABC/final'
output_sigma = os.path.join(path_base, '3_params_sigma', 'output',)
output_prod = os.path.join(path_base, '3_params_prod', 'output',)
output_both = os.path.join(path_base, '3_params_both', 'output',)
params_sigma = yaml.load(open(os.path.join(output_sigma, 'output_params.yml'), 'r'))
params_prod = yaml.load(open(os.path.join(output_prod, 'output_params.yml'), 'r'))
########################
filename_accepted_sigma = os.path.join(output_sigma, 'accepted_0.npz')
filename_accepted_prod = os.path.join(output_prod, 'accepted_0.npz')
accepted_sigma = np.load(filename_accepted_sigma)['C']
accepted_prod = np.load(filename_accepted_prod)['C']

C_limits_sigma = np.zeros((N_params, 2))
C_limits_prod = np.zeros((N_params, 2))
for i in range(N_params):
    C_limits_sigma[i] = [np.min(accepted_sigma[:, i]), np.max(accepted_sigma[:, i])]
    C_limits_prod[i] = [np.min(accepted_prod[:, i]), np.max(accepted_prod[:, i])]
C_limits_sigma[0, 1] = np.min([0.0, C_limits_sigma[0, 1]])
C_limits_prod[0, 1] = np.min([0.0, C_limits_prod[0, 1]])
print(C_limits_sigma)
print(C_limits_prod)

C_limits = np.zeros((N_params, 2))
for i in range(N_params):
    minimum = np.min([C_limits_sigma[i, 0], C_limits_prod[i, 0]])
    maximum = np.max([C_limits_sigma[i, 1], C_limits_prod[i, 1]])
    C_limits[i] = [minimum, maximum]
C_limits[0, 1] = np.min([0.0, C_limits[0, 1]])
print(C_limits)

max_value = 0.0
data = dict()
diagonal_sigma = dict()
diagonal_prod = dict()
diagonal_both = dict()
for i in range(N_params):
    for j in range(N_params):
        if i < j:   # sigma
            data[str(i)+str(j)] = np.loadtxt(os.path.join(output_sigma, 'conditional_smooth' + str(j) + str(i)))
            n_bins = data[str(i)+str(j)].shape[0]-1
            norm = np.sum(data[str(i)+str(j)])*(C_limits_sigma[j, 1] - C_limits_sigma[j, 0])*(C_limits_sigma[i, 1] - C_limits_sigma[i, 0]) / n_bins**2
            print('norm = ', norm, np.max(data[str(i)+str(j)]))
            data[str(i) + str(j)] /= norm
            max_value = max(max_value, np.max(data[str(i)+str(j)]))
            print(np.max(data[str(i) + str(j)]))
        if i > j: # prod
            data[str(i) + str(j)] = np.loadtxt(os.path.join(output_prod, 'conditional_smooth' + str(i) + str(j)))
            n_bins = data[str(i)+str(j)].shape[0]-1
            norm = np.sum(data[str(i)+str(j)])*(C_limits_prod[j, 1] - C_limits_prod[j, 0])*(C_limits_prod[i, 1] - C_limits_prod[i, 0]) / n_bins**2
            print('norm = ', norm, np.max(data[str(i)+str(j)]))
            data[str(i) + str(j)] /= norm
            max_value = max(max_value, np.max(data[str(i)+str(j)]))
            print('this', np.max(data[str(i) + str(j)]))
        if i == j:
            data_marg = np.loadtxt(os.path.join(output_sigma, 'marginal_smooth{}'.format(i)))
            norm = np.sum(data_marg[1]) * (data_marg[0, 1] - data_marg[0, 0])
            data_marg[1] /= norm
            diagonal_sigma[str(i)] = data_marg
            print('summ diag sigma= ', np.sum(diagonal_sigma[str(i)][1]) * (data_marg[0, 1] - data_marg[0, 0]))
            data_marg = np.loadtxt(os.path.join(output_prod, 'marginal_smooth{}'.format(i)))
            norm = np.sum(data_marg[1]) * (data_marg[0, 1] - data_marg[0, 0])
            data_marg[1] /= norm
            diagonal_prod[str(i)] = data_marg
            print('summ diag prod= ', np.sum(diagonal_prod[str(i)][1]) * (data_marg[0, 1] - data_marg[0, 0]))
            data_marg = np.loadtxt(os.path.join(output_both, 'marginal_smooth{}'.format(i)))
            norm = np.sum(data_marg[1]) * (data_marg[0, 1] - data_marg[0, 0])
            data_marg[1] /= norm
            diagonal_both[str(i)] = data_marg
            print('summ diag both= ', np.sum(diagonal_both[str(i)][1]) * (data_marg[0, 1] - data_marg[0, 0]))

print(max_value)
###################################################################################################################
cmap = cm.inferno  # define the colormap
cmaplist = [cmap(i) for i in reversed(range(cmap.N))]  # extract all colors from the map
gamma = 1/np.e   # exponential scale for map. 1 for linear
cmaplist[0] = 'white'   # force the first color entry to be white
cmap = LinearSegmentedColormap.from_list('Custom cmap', cmaplist)
###################################################################################################################
fig_width, fig_height = fig_size(width_column=double_column)
print('fig_height', fig_height, text_height*0.45, 0.6*fig_width)
fig = plt.figure(figsize=(fig_width, 0.45*text_height))
ticks = [0.06, 0.2, 0.2]
ax_diag = dict()
for i in range(N_params):
    for j in range(N_params):
        if i == j:
            ax_diag[i] = plt.subplot2grid((N_params, N_params), (i, i))
            # sigma
            c_final_smooth_sigma = np.loadtxt(os.path.join(output_sigma, 'C_final_smooth'))
            ax_diag[i].plot(diagonal_sigma[str(i)][0], diagonal_sigma[str(i)][1], 'b', label='marginal pdf sigma')
            labelsrt = 'maximum of \n joint pdf:'
            ax_diag[i].axvline(c_final_smooth_sigma[i], linestyle='--', color='b', label=labelsrt)
            # production
            c_final_smooth_prod = np.loadtxt(os.path.join(output_prod, 'C_final_smooth'))
            ax_diag[i].plot(diagonal_prod[str(i)][0], diagonal_prod[str(i)][1], 'r', label='marginal pdf')
            ax_diag[i].axvline(c_final_smooth_prod[i], linestyle='--', color='r', label=labelsrt)
            # both
            c_final_smooth_both = np.loadtxt(os.path.join(output_both, 'C_final_smooth'))
            ax_diag[i].plot(diagonal_both[str(i)][0], diagonal_both[str(i)][1], 'g', label='marginal pdf')
            ax_diag[i].axvline(c_final_smooth_both[i], linestyle='--', color='g', label=labelsrt)

            ax_diag[i].axis(xmin=C_limits[i, 0], xmax=C_limits[i, 1], ymin=0)
            ax_diag[i].tick_params(axis='both', which='major', pad=2)
            ax_diag[i].tick_params(axis='y', which='major', pad=0.8)
            ax_diag[i].xaxis.set_major_locator(ticker.MultipleLocator(ticks[i]))
            ax_diag[i].grid(True, linestyle='--', alpha=0.5)
            ax_diag[i].text(0.03, 0.87, '('+string.ascii_lowercase[i*N_params+j]+')',
                        transform=ax_diag[i].transAxes, size=10, weight='black')

            if i != (N_params - 1):
                ax_diag[i].xaxis.set_major_formatter(plt.NullFormatter())
            else:
                ax_diag[i].set_xlabel(params_names[i], labelpad=2.2)
            if i == 0:
                ax_diag[i].set_ylabel('pdf')
            # if i == 0:
            #     ax.legend(bbox_to_anchor=(5.25, 1.07), fancybox=True)
            #     textstr = '\n'.join((r'$C_1=%.3f$' % (c_final_smooth[0],),
            #                          r'$C_2=%.3f$' % (c_final_smooth[1],),
            #                          r'$C_3=%.3f$' % (c_final_smooth[2],)))
            #     ax.text(4.3, 0.35, textstr, transform=ax.transAxes, verticalalignment='top', linespacing=1.5)
            #     ax.text(4.4, -0.35, '2D marginal', transform=ax.transAxes, horizontalalignment='center',
            #             rotation=90, linespacing=1.5)
            #     ax.text(4.4, -1.48, 'conditional', transform=ax.transAxes, horizontalalignment='center',
            #             rotation=90, linespacing=1.5)

        if i < j:  # sigma
            ax = plt.subplot2grid((N_params, N_params), (i, j))
            ax.axis(xmin=C_limits[j, 0], xmax=C_limits[j, 1], ymin=C_limits[i, 0], ymax=C_limits[i, 1])

            ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.xaxis.set_major_locator(ticker.MultipleLocator(ticks[j]))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(ticks[i]))
            ax.tick_params(axis='both', which='major', pad=1.5)
            ax.yaxis.tick_right()
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.text(0.03, 0.87, '('+string.ascii_lowercase[i*N_params+j]+')',
                    transform=ax.transAxes, size=10, weight='black')

            if j != N_params-1:
                ax.yaxis.set_major_formatter(plt.NullFormatter())
            else:
                ax.set_ylabel(params_names[i])
                ax.yaxis.set_label_position("right")

            ext = (C_limits_sigma[j, 0], C_limits_sigma[j, 1], C_limits_sigma[i, 0], C_limits_sigma[i, 1])
            im = ax.imshow(data[str(i)+str(j)], origin='lower', cmap=cmap, aspect='auto',
                           extent=ext, norm=colors.PowerNorm(gamma=gamma), vmin=0, vmax=max_value)
            ax.axvline(c_final_smooth_sigma[j], linestyle='--', color='b')
            ax.axhline(c_final_smooth_sigma[i], linestyle='--', color='b')
        elif i > j: # production
            ax = plt.subplot2grid((N_params, N_params), (i, j))
            ax.axis(xmin=C_limits[j, 0], xmax=C_limits[j, 1], ymin=C_limits[i, 0], ymax=C_limits[i, 1])

            ax.tick_params(axis='both', which='major', pad=2)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(ticks[j]))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(ticks[i]))
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.text(0.03, 0.87, '('+string.ascii_lowercase[i*N_params+j]+')',
                    transform=ax.transAxes, size=10, weight='black')
            if j == 0:
                ax.set_ylabel(params_names[i])
            else:
                ax.yaxis.set_major_formatter(plt.NullFormatter())
            if i != (N_params - 1):
                ax.xaxis.set_major_formatter(plt.NullFormatter())
            else:
                ax.set_xlabel(params_names[j], labelpad=2.2)
            ext = (C_limits_prod[j, 0], C_limits_prod[j, 1], C_limits_prod[i, 0], C_limits_prod[i, 1])
            im_cond = ax.imshow(data[str(i) + str(j)].T, origin='lower', cmap=cmap,
                                aspect='auto', extent=ext, norm=colors.PowerNorm(gamma=gamma),
                                vmin=0, vmax=max_value)
            ax.axvline(c_final_smooth_prod[j], linestyle='--', color='r')
            ax.axhline(c_final_smooth_prod[i], linestyle='--', color='r')

ticks = np.arange(1, 100, 10)

# cax = plt.axes([0.18, 0.705, 0.01, 0.275])
# plt.colorbar(im, cax=cax, format=ticker.FormatStrFormatter('%.0e'), ticks=ticks1)
# cax.yaxis.set_ticks_position('left')
cax = plt.axes([0.81, 0.085, 0.01, 0.28])
plt.colorbar(im_cond, cax=cax)
# format=ticker.FormatStrFormatter('%.1e'),
# ticks=ticks
fig.subplots_adjust(left=0.2, right=0.8, wspace=0.1, hspace=0.1, bottom=0.085, top=0.98)
fig.savefig(os.path.join(path_base, 'conditional_combined_3'))
plt.close('all')