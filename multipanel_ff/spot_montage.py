#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:29:27 2017

@author: bernier2
"""

import argparse

import numpy as np
import h5py
from matplotlib import pyplot as plt


# Options
params = {'text.usetex': True,
          'font.size': 14,
          'font.family': 'mathrm',
          'text.latex.unicode': True,
          'pgf.texsystem': 'pdflatex'
          }
plt.rcParams.update(params)

plt.ion()


def montage(X, colormap=plt.cm.inferno, show_borders=True,
            title=None, xlabel=None, ylabel=None,
            threshold=None, filename=None):
    m, n, count = np.shape(X)
    img_data = np.log(X - np.min(X) + 1)
    if threshold is None:
        threshold = 0.
    else:
        threshold = np.log(threshold - np.min(X) + 1)
    mm = int(np.ceil(np.sqrt(count)))
    nn = mm
    M = np.zeros((mm * m, nn * n))

    # colormap
    colormap.set_under('b')

    fig, ax = plt.subplots()
    image_id = 0
    for j in range(mm):
        sliceM = j * m
        ax.plot()
        for k in range(nn):
            if image_id >= count:
                img = np.nan*np.ones((m, n))
            else:
                img = img_data[:, :, image_id]
            sliceN = k * n
            M[sliceM:sliceM + m, sliceN:sliceN + n] = img
            image_id += 1
    # M = np.sqrt(M + np.min(M))
    im = ax.imshow(M, cmap=colormap, vmin=threshold, interpolation='nearest')
    if show_borders:
        xs = np.vstack(
            [np.vstack([[n*i, n*i] for i in range(nn+1)]),
             np.tile([0, nn*n], (mm+1, 1))]
        )
        ys = np.vstack(
            [np.tile([0, mm*m], (nn+1, 1)),
             np.vstack([[m*i, m*i] for i in range(mm+1)])]
        )
        for xp, yp in zip(xs, ys):
            ax.plot(xp, yp, 'c:')
    if xlabel is None:
        ax.set_xlabel(r'$2\theta$', FontSize=14)
    else:
        ax.set_xlabel(xlabel, FontSize=14)
    if ylabel is None:
        ax.set_ylabel(r'$\eta$', FontSize=14)
    else:
        ax.set_ylabel(ylabel, FontSize=14)
    ax.axis('normal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    cbar_ax = fig.add_axes([0.875, 0.155, 0.025, 0.725])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r"$\ln(\mbox{intensity})", labelpad=5)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title, FontSize=18)
    if filename is not None:
        fig.savefig(filename, bbox_inches='tight', dpi=300)
    return M


def plot_gvec_from_hdf5(fname, gvec_id, threshold=0.):
    """
    """
    f = h5py.File(fname, 'r')

    for det_key, panel_data in f['reflection_data'].iteritems():
        for spot_id, spot_data in panel_data.iteritems():
            attrs = spot_data.attrs
            if attrs['hkl_id'] == gvec_id:
                # grab some data
                tth_crd = np.degrees(spot_data['tth_crd'])
                eta_crd = np.degrees(spot_data['eta_crd'])
                intensities = np.transpose(
                        np.array(spot_data['intensities']),
                        (1, 2, 0)
                )

                # make labels
                figname = r'Spot %d, ' % attrs['peak_id'] \
                    + r"detector '%s', " % det_key \
                    + r'({:^3} {:^3} {:^3})'.format(*attrs['hkl'])
                xlabel = r'$2\theta\in(%.3f, %.3f)$' \
                    % (tth_crd[0], tth_crd[-1])
                ylabel = r'$\eta\in(%.3f, %.3f)$' \
                    % (eta_crd[0], eta_crd[-1])

                # make montage
                montage(intensities, title=figname,
                        xlabel=xlabel, ylabel=ylabel,
                        threshold=threshold)
                pass
            pass
        pass
    f.close()
    return


# =============================================================================
# %% CMD LINE HOOK
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Montage of spot data for a specifed G-vector family")

    parser.add_argument('hdf5_archive',
                        help="hdf5 archive filename",
                        type=str)
    parser.add_argument('gvec_id',
                        help="unique G-vector ID from PlaneData",
                        type=int)

    parser.add_argument('-t', '--threshold',
                        help="intensity threshold",
                        type=float, default=0.)

    args = parser.parse_args()

    h5file = args.hdf5_archive
    hklid = args.gvec_id
    threshold = args.threshold

    plot_gvec_from_hdf5(h5file, hklid, threshold=threshold)