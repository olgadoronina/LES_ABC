import os
import numpy as np
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# plt.style.use('dark_background')

fig_width_pt = 1.5*246.0  # Get this from LaTeX using "The column width is: \the\columnwidth \\"
inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean       # height in inches
fig_size = [fig_width, fig_height]

# mpl.rcParams['figure.figsize'] = 6.5, 2.2
# plt.rcParams['figure.autolayout'] = True

mpl.rcParams['font.size'] = 10
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


path = '../ABC/comparison/'

# colors = ['', 'cyan', 'yellow', 'green']
colors = ['k', 'b', 'r']
# colors = np.array([[27, 158, 119], [217, 95, 2], [117, 112, 179]])/255

fig, axarr = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(fig_width, 0.45*fig_width))
titles = [r'$\sigma_{11}$', r'$\sigma_{12}$', r'$\sigma_{13}$', r'$\sigma_{ij}S_{ij}$']
y_true = dict()
y_max_joint = dict()
x = np.loadtxt(os.path.join(path, 'sum_stat_bins'))[0]
for ind, key in enumerate(['uu', 'uv', 'uw']):
    print(ind, key)
    # Plot true pdf
    y_true[key] = np.loadtxt(os.path.join(path, 'sum_stat_true'))[ind]
    axarr[ind].plot(x, y_true[key], colors[0], linewidth=1, label='true')
    # plot max smooth
    if os.path.isfile(os.path.join(path, '3_param_sigma')):
        y_max_joint[key] = np.loadtxt(os.path.join(path, '3_param_sigma'))[ind]
        axarr[ind].plot(x, y_max_joint[key], ':',  color=colors[1], linewidth=1, label='3 param sigma')
    if os.path.isfile(os.path.join(path, '4_param_sigma')):
        y_max_joint[key] = np.loadtxt(os.path.join(path, '4_param_sigma'))[ind]
        axarr[ind].plot(x, y_max_joint[key], ':',  color=colors[2], linewidth=1, label='4 param sigma')
    if os.path.isfile(os.path.join(path, '3_param_production')):
        y_max_joint[key] = np.loadtxt(os.path.join(path, '3_param_production'))[ind]
        axarr[ind].plot(x, y_max_joint[key], '--', color=colors[1], linewidth=1, label='3 param production')
    if os.path.isfile(os.path.join(path, '4_param_production')):
        y_max_joint[key] = np.loadtxt(os.path.join(path, '4_param_production'))[ind]
        axarr[ind].plot(x, y_max_joint[key], '--', color=colors[2], linewidth=1, label='4 param production')
    axarr[ind].set_xlabel(titles[ind], labelpad=2)
    axarr[ind].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    axarr[ind].xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    axarr[ind].set_xlim([-0.3, 0.3])
    print('done')
axarr[0].set_ylabel('ln(pdf)', labelpad=2)
axarr[0].yaxis.set_major_locator(ticker.MultipleLocator(2))

x = np.loadtxt(os.path.join(path, 'sum_stat_bins'))[1]
y_true = np.loadtxt(os.path.join(path, 'sum_stat_true'))[3]
axarr[3].plot(x, y_true, colors[0], linewidth=1, label='true')
if os.path.isfile(os.path.join(path,  '3_param_sigma')):
    y_max_joint = np.loadtxt(os.path.join(path, '3_param_sigma'))[3]
    axarr[3].plot(x, y_max_joint, ':',  color=colors[1], linewidth=1)
if os.path.isfile(os.path.join(path, '4_param_sigma')):
    y_max_joint = np.loadtxt(os.path.join(path, '4_param_sigma'))[3]
    axarr[3].plot(x, y_max_joint, ':', color=colors[2], linewidth=1)

if os.path.isfile(os.path.join(path, '3_param_production')):
    y_max_joint = np.loadtxt(os.path.join(path, '3_param_production'))[3]
    axarr[3].plot(x, y_max_joint, '--', color=colors[1], linewidth=1)
if os.path.isfile(os.path.join(path, '4_param_production')):
    y_max_joint = np.loadtxt(os.path.join(path, '4_param_production'))[3]
    axarr[3].plot(x, y_max_joint, '--', color=colors[2], linewidth=1)
axarr[3].set_xlabel(titles[3], labelpad=2)
axarr[3].set_ylim(bottom=-7)
axarr[3].set_xlim([-5,5])
axarr[3].xaxis.set_major_locator(ticker.MultipleLocator(2))
fig.subplots_adjust(left=0.12, right=0.98, wspace=0.1, bottom=0.22, top=0.75)


custom_lines = [Line2D([0], [0], color=colors[0], lw=1),
                Line2D([0], [0], color=colors[1], linestyle='-', lw=1),
                Line2D([0], [0], color=colors[2], linestyle='-', lw=1)]
axarr[1].legend(custom_lines, ['true data', '3 parameters', '4 parameters'], loc='upper center', bbox_to_anchor=(0.9, 1.3), frameon=False,
                fancybox=False, shadow=False, ncol=3)
# custom_lines2 = [Line2D([0], [0], color='w', linestyle='--', lw=1),
#                 Line2D([0], [0], color='w', linestyle=':', lw=1)]
# axarr[2].legend(custom_lines2, ['production', 'sigma'], loc='upper center', bbox_to_anchor=(-0.3, 1.29), frameon=False,
#                 fancybox=False, shadow=False, ncol=3)
fig.savefig(os.path.join(path, 'compare_sum_stat'))
plt.close('all')