import gc
from math import *

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def imagesc(Arrays, map_bounds, name=None):
    cmap = plt.cm.jet  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]  # extract all colors from the .jet map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)  # create the new map
    norm = mpl.colors.BoundaryNorm(map_bounds, cmap.N)

    # fig = plt.figure()
    axis = [0, 2 * pi, 0, 2 * pi]
    titles = ['DNS data', 'LES filter', 'Test filter']
    if len(Arrays) > 1:
        fig, axes = plt.subplots(nrows=1, ncols=len(Arrays), figsize=(12, 4))
        k = 0
        for ax in axes.flat:
            im = ax.imshow(Arrays[k].T, origin='lower', cmap=cmap, norm=norm, interpolation="nearest", extent=axis)
            ax.set_title(titles[k])
            k += 1
        cbar_ax = fig.add_axes([0.85, 0.17 , 0.015, 0.66])  # ([0.85, 0.15, 0.05, 0.7])
        fig.subplots_adjust(right=0.8)
        # fig.tight_layout()
        fig.colorbar(im, cax=cbar_ax, ax=axes.ravel().tolist())
    else:
        fig = plt.figure(figsize=(12, 10))
        ax = plt.gca()
        im = ax.imshow(Arrays[0].T, origin='lower', cmap=cmap, norm=norm, interpolation="nearest")
        fig.tight_layout()
        plt.colorbar(im, fraction=0.05, pad=0.04)

    fig1 = plt.gcf()
    plt.show()
    if name:
        fig1.savefig(name + '.eps')
    del ax, im, fig, fig1, cmap
    gc.collect()


def histogram(field, bins, pdf=None, label=None, log=False):
    plt.figure(figsize=(8, 8))
    plt.hist(field, bins=bins, normed=1, alpha=0.4)

    # h, edges = np.histogram(field, bins=bins, range=[-2,2], normed=1)
    if pdf:
        x = np.linspace(min(field), max(field), 100)
        mu = np.mean(field)
        sigma = np.std(field)
        plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r--', linewidth=3, label='Gaussian')
        plt.legend(loc=0)
    if log:
        plt.yscale('log', nonposy='clip')
    plt.xlabel(label)
    plt.ylabel('pdf(' + label + ')')
    plt.axis(xmin=np.min(field), xmax=np.max(field))  # xmax=4xmax = np.max(field)
    plt.show()

    # return h, edges

def T_TEST(T_TEST):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(10, 4))
    titles = [r'$T_{11}$', r'$T_{12}$', r'$T_{13}$']
    ax1.hist(T_TEST['uu'].flatten(), bins=100, normed=1, alpha=0.4)
    ax2.hist(T_TEST['uv'].flatten(), bins=100, normed=1, alpha=0.4)
    ax3.hist(T_TEST['uw'].flatten(), bins=100, normed=1, alpha=0.4)

    for ind, ax in enumerate([ax1, ax2, ax3]):
        ax.set_xlabel(titles[ind])
    ax1.axis(xmin=-1.1, xmax=1.1, ymin=1e-5)
    ax1.set_ylabel('pdf')
    ax3.set_yscale('log', nonposy='clip')
    fig.tight_layout()
    # plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    plt.show()

def TS(TS):
    plt.figure(figsize=(5, 5))
    plt.hist(TS.flatten(), bins=500, normed=1, alpha=0.4)
    plt.yscale('log', nonposy='clip')
    plt.xlabel(r'$T_{ij}S^T_{ij}$')
    plt.ylabel(r'pdf($T_{ij}S^T_{ij}$)')
    plt.axis(xmin=-5, xmax=4, ymin=1e-5)
    plt.show()