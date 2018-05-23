import os
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from params import output

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
mpl.rcParams['xtick.minor.size'] = 2
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.width'] = 0.5
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['ytick.minor.size'] = 2
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.width'] = 0.5
plt.rcParams['axes.linewidth'] = 1


def plot_sum_stat(name):

    sum_stat = np.loadtxt(os.path.join(output['output_path'], 'sum_stat_true'))
    bins = np.loadtxt(os.path.join(output['output_path'], 'sum_stat_bins'))

    if name == 'sigma_pdf_log':
        fig, axarr = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(6.5, 2.4))
        titles = [r'$\sigma_{11}$', r'$\sigma_{12}$', r'$\sigma_{13}$']

        for i in range(len(sum_stat)):
            axarr[i].plot(bins, sum_stat[i], 'r', linewidth=2)
            axarr[i].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
            axarr[i].xaxis.set_major_locator(ticker.MultipleLocator(0.45))
            axarr[i].set_xlabel(titles[i])

        axarr[0].axis(xmin=-0.5, xmax=0.5)
        axarr[0].set_ylabel('ln(pdf)')
        # axarr[0].set_yscale('log')
        fig.subplots_adjust(left=0.1, right=0.95, wspace=0.1, bottom=0.2, top=0.9)

    elif name == 'production_pdf_log':
        fig = plt.figure(figsize=(4, 3))
        ax = plt.gca()
        title = r'$\widetilde{P}$'

        ax.plot(bins, sum_stat, 'r', linewidth=2)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.axis(xmin=-5, xmax=5)
        ax.set_xlabel(title)
        ax.set_ylabel('ln(pdf)')
        # ax.set_yscale('log')
        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)

    fig.savefig(os.path.join(output['plot_path'], name))
    plt.close('all')

plot_sum_stat('production_pdf_log')