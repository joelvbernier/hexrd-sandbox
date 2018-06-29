#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 18:51:06 2018

@author: rachel
"""

import os
import time

import yaml

import numpy as np

import glob

try:
    import dill as cpl
except(ImportError):
    import cPickle as cpl

from hexrd.xrd import material
from hexrd import config

from hexrd import imageseries
from hexrd.imageseries.omega import OmegaImageSeries

from hexrd import instrument

from matplotlib import pyplot as plt

import hexrd.fitting.peakfunctions as pkfuncs
import scipy.optimize as optimize
from scipy.interpolate import interp1d


# =============================================================================
# %% Functions
# =============================================================================
def make_matl(mat_name, sgnum, lparms, hkl_ssq_max=100):
    matl = material.Material(mat_name)
    matl.sgnum = sgnum
    matl.latticeParameters = lparms
    matl.hklMax = hkl_ssq_max

    nhkls = len(matl.planeData.exclusions)
    matl.planeData.set_exclusions(np.zeros(nhkls, dtype=bool))
    return matl


# Multipeak Kludge
def fit_pk_obj_1d_mpeak(p, x, f0, pktype, num_pks):
    f = np.zeros(len(x))
    p = np.reshape(p, [num_pks, p.shape[0]/num_pks])
    for ii in np.arange(num_pks):
        if pktype == 'gaussian':
            f += pkfuncs._gaussian1d_no_bg(p[ii], x)
        elif pktype == 'lorentzian':
            f += pkfuncs._lorentzian1d_no_bg(p[ii], x)
        elif pktype == 'pvoigt':
            f += pkfuncs._pvoigt1d_no_bg(p[ii], x)
        elif pktype == 'split_pvoigt':
            f += pkfuncs._split_pvoigt1d_no_bg(p[ii], x)
    return f - f0


# plane data
def load_pdata(cpkl, key):
    with file(cpkl, "r") as matf:
        mat_list = cpl.load(matf)
    return dict(zip([i.name for i in mat_list], mat_list))[key].planeData


# images
def load_images(yml):
    return imageseries.open(yml, format="frame-cache", style="npz")


# instrument
def load_instrument(yml):
    with file(yml, 'r') as f:
        icfg = yaml.load(f)
    return instrument.HEDMInstrument(instrument_config=icfg)


# =============================================================================
# %% User input
# =============================================================================

cfg_filename = 'LSHR8_S0.yml'
samp_name = 'LSHR8_S0'
scan_number = 0

data_dir = os.getcwd()

fc_stem = "%s_%06d-fc_%%s.npz" % (samp_name, scan_number)

tth_tol = 0.5  # width around ring, degrees
eta_tol = 5.0  # azimuthal width, degrees
npdiv = 2  # pixel subsampling level
# =============================================================================
# %% Initialization
# =============================================================================

cfg = config.open(cfg_filename)[0]

analysis_id = '%s_%s' % (
    cfg.analysis_name.strip().replace(' ', '-'),
    cfg.material.active.strip().replace(' ', '-'),
    )

# handle max tth
max_tth = cfg.fit_grains.tth_max
if max_tth:
    if type(cfg.fit_grains.tth_max) != bool:
        max_tth = np.radians(float(max_tth))
else:
    max_tth = None

# load plane data
pd = load_pdata(cfg.material.definitions, cfg.material.active)
pd.tThMax = max_tth
if tth_tol is not None:
    pd.tThWidth = np.radians(tth_tol)

# load instrument
instr = load_instrument(cfg.instrument.parameters)
det_keys = instr.detectors.keys()

# =============================================================================
# %% Open ImageSeries
# =============================================================================
img_start = time.clock()

imsd = dict.fromkeys(det_keys)
for det_key in det_keys:
    fc_file = sorted(
        glob.glob(
            os.path.join(
                data_dir,
                fc_stem % det_key
            )
        )
    )

    if len(fc_file) != 1:
        raise(RuntimeError, 'cache file not found, or multiple found')
    else:
        ims = load_images(fc_file[0])
        imsd[det_key] = OmegaImageSeries(ims)

img_elapsed = time.clock() - img_start
print("imageseries.open took %.2f seconds" % img_elapsed)


# =============================================================================
# %% Max across frames (create "powder" pattern)
# =============================================================================
max_start = time.clock()

max_frames = dict.fromkeys(det_keys)
for det_key in det_keys:
    max_frames[det_key] = imageseries.stats.average(imsd[det_key])

max_elapsed = time.clock() - max_start
print("Max across frame calculation took %.2f seconds; %.2e per frame"
      % (max_elapsed, max_elapsed/float(len(imsd[det_key]))))

# =============================================================================
# %% Set pixel size and such
# =============================================================================
print("current tth width (from planeData): %f", np.degrees(pd.tThWidth))
tth_del = pd.tThWidth
tth_avg = np.average(pd.getTTh())
tth_lo = pd.getTTh()[0] - tth_del
tth_hi = pd.getTTh()[-1] + tth_del

nrings = len(pd.getMergedRanges()[0])

# !!! this assumes tacitly all panels are the same and at the same distance
panel_id = det_keys[0]
d = instr.detectors[panel_id]
pangs, pxys = d.make_powder_rings(pd)
aps = d.angularPixelSize(pxys[0])

print("min angular pixel sizes: %.4f, %.4f"
      % (np.degrees(np.min(aps[:, 0])), np.degrees(np.min(aps[:, 1]))))


# %% set from looking at GUI
lines = instr.extract_line_positions(
    pd, max_frames, eta_tol=eta_tol, npdiv=npdiv,
    collapse_eta=True, collapse_tth=False, do_interpolation=True)

# the lines object goes
# tth_edges = [panel][ring][patch][image][0]
# ref_eta = [panel][ring][patch][image][1]
#
# if you want to collapse to lineouts...

# %%
ring_data_p = dict.fromkeys(instr.detectors)
for det_key in det_keys:
    ring_tth = []
    ring_int = []
    for i_ring, ring_data in enumerate(lines[det_key]):
        this_tth = []
        this_int = []
        for i_azim, patch_data in enumerate(ring_data):
            this_tth.append(
                np.average(
                    np.vstack(
                        [patch_data[0][0][:-1],
                         patch_data[0][0][1:]]
                    ), axis=0)
            )
            this_int.append(np.array(patch_data[1][0]).flatten())
            pass

        # loop to find tth ranges
        tth_range = [
            np.min([np.min(tth) for tth in this_tth]),
            np.max([np.max(tth) for tth in this_tth])
        ]

        # loop to find delta tth vals
        del_tth = np.min([i[1] - i[0] for i in this_tth])

        new_del_tth = int(np.ceil((tth_range[-1] - tth_range[0])/del_tth))

        new_tth = np.linspace(tth_range[0], tth_range[1], num=new_del_tth)
        new_int = np.zeros_like(new_tth)
        for x, y in zip(this_tth, this_int):
            ifunc = interp1d(x, y, kind='linear', fill_value='extrapolate')
            new_int += ifunc(new_tth)
            pass
        ring_tth.append(new_tth)
        ring_int.append(new_int)
        pass
    ring_data_p[det_key] = [
        np.stack([i, j], axis=0) for i, j in zip(ring_tth, ring_int)
    ]
    pass

# %%
ring_tth_all = []
ring_int_all = []
for ir in range(nrings):
    this_tth = [pr[ir][0] for pr in ring_data_p.itervalues()]
    this_int = [pr[ir][1] for pr in ring_data_p.itervalues()]

    # loop to find tth ranges
    tth_range = [
        np.min([np.min(tth) for tth in this_tth]),
        np.max([np.max(tth) for tth in this_tth])
    ]

    # loop to find delta tth vals
    del_tth = np.min([i[1] - i[0] for i in this_tth])

    new_del_tth = int(np.ceil((tth_range[-1] - tth_range[0])/del_tth))

    new_tth = np.linspace(tth_range[0], tth_range[1], num=new_del_tth)
    new_int = np.zeros_like(new_tth)
    for x, y in zip(this_tth, this_int):
        ifunc = interp1d(x, y, kind='linear', fill_value='extrapolate')
        new_int += ifunc(new_tth)
        pass
    ring_tth_all.append(new_tth)
    ring_int_all.append(new_int)

fig, ax = plt.subplots()
for tth, inten in zip(ring_tth_all, ring_int_all):
    ax.plot(np.degrees(tth), inten)
    ax.set_xlabel(r'$2\theta$')
    ax.set_ylabel(r'Intensity')
ax.grid(True)
fig.suptitle(r'%s' % samp_name)

# =============================================================================
# %%
# =============================================================================
"""
fig, ax = plt.subplots(2, 1, sharex=True, sharey=False)
ax[0].imshow(img3.reshape(neta, ntth),
             interpolation='nearest',
             cmap=cm.plasma, vmax=None,
             extent=extent,
             origin='lower')
lineout = np.sqrt(np.sum(img2, axis=0))
ax[1].plot(angpts[1][0, :], lineout)
ax[0].axis('tight')
ax[0].grid(True)
ax[1].grid(True)
ax[0].set_ylabel(r'$\eta$ [deg]', size=18)
ax[1].set_xlabel(r'$2\theta$ [deg]', size=18)
ax[1].set_ylabel(r'$\sqrt{Int}$ (arb.)', size=18)

plt.show()

#%%
#plt.close('all')

num_tth=len(pd.getTTh())

x=angpts[1][0, :]
f=lineout
pktype='pvoigt'
num_pks=num_tth

ftol=1e-6
xtol=1e-6

fitArgs=(x,f,pktype,num_pks)

tth=matl.planeData.getTTh()*180./np.pi


p0 = np.zeros([num_tth, 4])

for ii in np.arange(num_tth):
    pt = np.argmin(np.abs(x-tth[ii]))
    p0[ii,:] = [f[pt], tth[ii], 0.1, 0.5]

p, outflag = optimize.leastsq(fit_pk_obj_1d_mpeak, p0, args=fitArgs,ftol=ftol,xtol=xtol)

p=np.reshape(p,[num_pks,p.shape[0]/num_pks])
f_fit=np.zeros(len(x))

for ii in np.arange(num_pks):
    f_fit=f_fit+pkfuncs._pvoigt1d_no_bg(p[ii],x)


#plt.plot(x,f,'x')
#plt.hold('true')
#plt.plot(x,f_fit)
ax[1].plot(x, f_fit, 'm+', ms=1)

#%%
fit_tths = p[:, 1]
fit_dsps = 0.5*wlen/np.sin(0.5*np.radians(fit_tths))
nrml_strains = fit_dsps/pd.getPlaneSpacings() - 1.

print nrml_strains
print "avg normal strain: %.3e" %np.average(nrml_strains)
"""


