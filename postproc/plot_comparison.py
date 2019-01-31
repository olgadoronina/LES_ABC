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


path = '../ABC/final/comparison/'

# colors = ['', 'cyan', 'yellow', 'green']
colors = ['k', 'b', 'r']
# colors = np.array([[27, 158, 119], [217, 95, 2], [117, 112, 179]])/255

fig, axarr = plt.subplots(nrows=3, ncols=4, sharey=True, figsize=(fig_width, fig_width))
labels = [r'$\sigma_{11}$', r'$\sigma_{12}$', r'$\sigma_{13}$', r'$\sigma_{ij}S_{ij}$']
titles = ['Trained on sigma', 'Trained on production', 'Trained on combination of sigma and production']
linestyles = ['-', ':', '--']
y_true = dict()
y_max_joint = dict()
x = np.loadtxt(os.path.join(path, 'sum_stat_bins'))[0]
for ind, key in enumerate(['uu', 'uv', 'uw']):
    print(ind, key)
    # Plot true pdf
    y_true[key] = np.loadtxt(os.path.join(path, 'sum_stat_true'))[ind]

    # plot max smooth
    if os.path.isfile(os.path.join(path, '3_param_sigma')):
        y_max_joint[key] = np.loadtxt(os.path.join(path, '3_param_sigma'))[ind]
        axarr[0, ind].plot(x, y_max_joint[key], linestyles[1],  color=colors[1], linewidth=1, label='3 param sigma')
    if os.path.isfile(os.path.join(path, '4_param_sigma')):
        y_max_joint[key] = np.loadtxt(os.path.join(path, '4_param_sigma'))[ind]
        axarr[0, ind].plot(x, y_max_joint[key], linestyles[2],  color=colors[2], linewidth=1, label='4 param sigma')
    if os.path.isfile(os.path.join(path, '3_param_production')):
        y_max_joint[key] = np.loadtxt(os.path.join(path, '3_param_production'))[ind]
        axarr[1, ind].plot(x, y_max_joint[key], linestyles[1], color=colors[1], linewidth=1, label='3 param production')
    if os.path.isfile(os.path.join(path, '4_param_production')):
        y_max_joint[key] = np.loadtxt(os.path.join(path, '4_param_production'))[ind]
        axarr[1, ind].plot(x, y_max_joint[key], linestyles[2], color=colors[2], linewidth=1, label='4 param production')
    if os.path.isfile(os.path.join(path, '3_param_production')):
        y_max_joint[key] = np.loadtxt(os.path.join(path, '3_param_both'))[ind]
        axarr[2, ind].plot(x, y_max_joint[key], linestyles[1], color=colors[1], linewidth=1, label='3 param sigma+production')
    for i in range(3):
        axarr[i, ind].plot(x, y_true[key], colors[0], linewidth=1, label='true data')
        axarr[i, ind].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        axarr[i, ind].xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        axarr[i, ind].set_xlim([-0.3, 0.3])
        if i < 2:
            axarr[i, ind].xaxis.set_major_formatter(plt.NullFormatter())
    print('done')

x = np.loadtxt(os.path.join(path, 'sum_stat_bins'))[1]
y_true = np.loadtxt(os.path.join(path, 'sum_stat_true'))[3]

if os.path.isfile(os.path.join(path,  '3_param_sigma')):
    y_max_joint = np.loadtxt(os.path.join(path, '3_param_sigma'))[3]
    axarr[0, 3].plot(x, y_max_joint, linestyles[1],  color=colors[1], linewidth=1)
if os.path.isfile(os.path.join(path, '4_param_sigma')):
    y_max_joint = np.loadtxt(os.path.join(path, '4_param_sigma'))[3]
    axarr[0, 3].plot(x, y_max_joint, linestyles[2], color=colors[2], linewidth=1)
if os.path.isfile(os.path.join(path, '3_param_production')):
    y_max_joint = np.loadtxt(os.path.join(path, '3_param_production'))[3]
    axarr[1, 3].plot(x, y_max_joint, linestyles[1], color=colors[1], linewidth=1)
if os.path.isfile(os.path.join(path, '4_param_production')):
    y_max_joint = np.loadtxt(os.path.join(path, '4_param_production'))[3]
    axarr[1, 3].plot(x, y_max_joint, linestyles[2], color=colors[2], linewidth=1)
if os.path.isfile(os.path.join(path, '3_param_production')):
    y_max_joint = np.loadtxt(os.path.join(path, '3_param_both'))[3]
    axarr[2, 3].plot(x, y_max_joint, linestyles[1], color=colors[1], linewidth=1)
for i in range(3):
    axarr[i, 3].plot(x, y_true, colors[0], linewidth=1, label='true')
    axarr[i, 3].set_ylim(bottom=-7)
    axarr[i, 3].set_xlim([-5, 5])
    axarr[i, 3].xaxis.set_major_locator(ticker.MultipleLocator(2))
    # axarr[i, 3].set_title(titles[i])
    if i < 2:
        axarr[i, 3].xaxis.set_major_formatter(plt.NullFormatter())


for i in range(4):
    axarr[2, i].set_xlabel(labels[i], labelpad=2)
for i in range(3):
    axarr[i, 0].set_ylabel('ln(pdf)', labelpad=2)

fig.subplots_adjust(left=0.1, right=0.98, wspace=0.1, hspace=0.2, bottom=0.08, top=0.9)


custom_lines = [Line2D([0], [0], color=colors[0], lw=1),
                Line2D([0], [0], color=colors[1], linestyle=':', lw=1),
                Line2D([0], [0], color=colors[2], linestyle='--', lw=1)]
axarr[0, 1].legend(custom_lines, ['true data', '3 parameters', '4 parameters'], loc='upper center',
                   bbox_to_anchor=(0.95, 1.45), frameon=True,
                   fancybox=True, shadow=False, ncol=3)
# custom_lines2 = [Line2D([0], [0], color='w', linestyle='--', lw=1),
#                 Line2D([0], [0], color='w', linestyle=':', lw=1)]
# axarr[2].legend(custom_lines2, ['production', 'sigma'], loc='upper center', bbox_to_anchor=(-0.3, 1.29), frameon=False,
#                 fancybox=False, shadow=False, ncol=3)
axarr[0, 0].text(0.95, 6.0, titles[0], verticalalignment='top', horizontalalignment='center')
axarr[1, 0].text(0.95, 6.0, titles[1], verticalalignment='top', horizontalalignment='center')
axarr[2, 0].text(0.95, 6.0, titles[2], verticalalignment='top', horizontalalignment='center')

# ax.text(4.4, -0.35, '2D marginal', transform=ax.transAxes, horizontalalignment='center',
#         rotation=90, linespacing=1.5)
# ax.text(4.4, -1.48, 'condisional', transform=ax.transAxes, horizontalalignment='center',
#         rotation=90, linespacing=1.5)

fig.savefig(os.path.join(path, 'compare_sum_stat'))
plt.close('all')