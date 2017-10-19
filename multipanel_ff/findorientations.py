#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 19:04:10 2017

@author: bernier2
"""
from __future__ import print_function

import os

import glob

import multiprocessing

import numpy as np

from scipy import ndimage

import timeit

try:
    import dill as cpl
except(ImportError):
    import cPickle as cpl

import yaml

from hexrd import constants as cnst
from hexrd import config
from hexrd import imageseries
from hexrd.imageseries.omega import OmegaImageSeries
from hexrd import instrument
from hexrd.findorientations import \
    generate_orientation_fibers, \
    run_cluster
from hexrd.xrd import transforms_CAPI as xfcapi
from hexrd.xrd import indexer
from matplotlib import pyplot as plt


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


# %%
# =============================================================================
# START USER INPUT
# =============================================================================

# cfg file -- currently ignores image_series block
cfg_filename = 'ruby_config.yml'
samp_name = 'RUBY'
scan_number = 0
data_dir = os.getcwd()

fc_stem = "%s_%04d-fc_%%s*.npz" % (samp_name, scan_number)

make_max_frames = False
use_direct_search = False

# for clustering neighborhood
# FIXME
min_samples = 15

# =============================================================================
# END USER INPUT
# =============================================================================

# %%
cfg = config.open(cfg_filename)[0]

analysis_id = '%s_%s' % (
    cfg.analysis_name.strip().replace(' ', '-'),
    cfg.material.active.strip().replace(' ', '-'),
    )

active_hkls = cfg.find_orientations.orientation_maps.active_hkls
if active_hkls == 'all':
    active_hkls = None

max_tth = cfg.fit_grains.tth_max
try:
    max_tth = np.degrees(float(max_tth))
except(ValueError):
    max_tth = None

# load plane data
plane_data = load_pdata(cfg.material.definitions, cfg.material.active)
plane_data.tThMax = max_tth

# load instrument
instr = load_instrument(cfg.instrument.parameters)
det_keys = instr.detectors.keys()

# grab eta ranges
eta_ranges = cfg.find_orientations.eta.range

# for indexing
build_map_threshold = cfg.find_orientations.orientation_maps.threshold

on_map_threshold = cfg.find_orientations.threshold
fiber_ndiv = cfg.find_orientations.seed_search.fiber_ndiv
fiber_seeds = cfg.find_orientations.seed_search.hkl_seeds

tth_tol = np.degrees(plane_data.tThWidth)
eta_tol = cfg.find_orientations.eta.tolerance
ome_tol = cfg.find_orientations.omega.tolerance
# omega period...
# QUESTION: necessary???
ome_period = np.radians(cfg.find_orientations.omega.period)

npdiv = cfg.fit_grains.npdiv

compl_thresh = cfg.find_orientations.clustering.completeness
cl_radius = cfg.find_orientations.clustering.radius

# %%

imsd = dict.fromkeys(det_keys)
for det_key in det_keys:
    fc_file = glob.glob(
        os.path.join(
            data_dir,
            fc_stem % det_key
        )
    )
    if len(fc_file) != 1:
        raise(RuntimeError, 'cache file not found, or multiple found')
    else:
        ims = load_images(fc_file[0])
        imsd[det_key] = OmegaImageSeries(ims)


if make_max_frames:
    max_frames_output_name = os.path.join(
        data_dir,
        "%s_%d-maxframes.hdf5" % (samp_name, scan_number)
    )

    if os.path.exists(max_frames_output_name):
        os.remove(max_frames_output_name)

    max_frames = dict.fromkeys(det_keys)
    for det_key in det_keys:
        max_frames[det_key] = imageseries.stats.max(imsd[det_key])

    ims_out = imageseries.open(
            None, 'array',
            data=np.array([max_frames[i] for i in max_frames]),
            meta={'panels': max_frames.keys()}
        )
    imageseries.write(
            ims_out, max_frames_output_name,
            'hdf5', path='/imageseries'
        )
# %%

print("INFO:\tbuilding eta_ome maps")
start = timeit.default_timer()

# make eta_ome maps
eta_ome = instrument.GenerateEtaOmeMaps(
    imsd, instr, plane_data,
    active_hkls=active_hkls, threshold=build_map_threshold,
    ome_period=cfg.find_orientations.omega.period)

print("INFO:\t\t...took %f seconds" % (timeit.default_timer() - start))

# save them
eta_ome.save(analysis_id + "_maps.npz")

# %%
cmap = plt.cm.hot
cmap.set_under('b')

fig, ax = plt.subplots()
i_ring = 0
this_map_f = -ndimage.filters.gaussian_laplace(
    eta_ome.dataStore[i_ring], 1.0,
)
ax.imshow(this_map_f, interpolation='nearest',
          vmin=on_map_threshold, vmax=None, cmap=cmap)
labels, num_spots = ndimage.label(
    this_map_f > on_map_threshold, ndimage.generate_binary_structure(2, 1)
)
coms = np.atleast_2d(
    ndimage.center_of_mass(
        eta_ome.dataStore[i_ring], labels=labels, index=range(1, num_spots+1)
    )
)
ax.hold(True)
ax.plot(coms[:, 1], coms[:, 0], 'm+', ms=12)
ax.axis('tight')
# %%
# =============================================================================
# SEARCH SPACE GENERATION
# =============================================================================
ncpus = cfg.multiprocessing

print("INFO:\tgenerating search quaternion list using %d processes" % ncpus)
start = timeit.default_timer()

qfib = generate_orientation_fibers(
    eta_ome, instr.chi, on_map_threshold,
    fiber_seeds, fiber_ndiv,
    ncpus=ncpus)
print("INFO:\t\t...took %f seconds" % (timeit.default_timer() - start))
print("INFO: will test %d quaternions using %d processes"
      % (qfib.shape[1], ncpus))

# %%
# =============================================================================
# ORIENTATION SCORING
# =============================================================================

if use_direct_search:
    def test_orientation_FF_init(params):
        global paramMP
        paramMP = params

    def test_orientation_FF_reduced(quat):
        """
        input parameters are [
        plane_data, instrument, imgser_dict,
        tth_tol, eta_tol, ome_tol, npdiv, threshold
        ]
        """
        plane_data = paramMP['plane_data']
        instrument = paramMP['instrument']
        imgser_dict = paramMP['imgser_dict']
        tth_tol = paramMP['tth_tol']
        eta_tol = paramMP['eta_tol']
        ome_tol = paramMP['ome_tol']
        npdiv = paramMP['npdiv']
        threshold = paramMP['threshold']

        phi = 2*np.arccos(quat[0])
        n = xfcapi.unitRowVector(quat[1:])
        grain_params = np.hstack([
            phi*n, cnst.zeros_3, cnst.identity_6x1,
        ])

        compl, scrap = instrument.pull_spots(
            plane_data, grain_params, imgser_dict,
            tth_tol=tth_tol, eta_tol=eta_tol, ome_tol=ome_tol,
            npdiv=npdiv, threshold=threshold,
            eta_ranges=None, ome_period=(-np.pi, np.pi),
            check_only=True)

        return sum(compl)/float(len(compl))

    params = dict(
            plane_data=plane_data,
            instrument=instr,
            imgser_dict=imsd,
            tth_tol=tth_tol,
            eta_tol=eta_tol,
            ome_tol=ome_tol,
            npdiv=npdiv,
            threshold=cfg.fit_grains.threshold)

    print("INFO:\tusing direct seach")
    pool = multiprocessing.Pool(ncpus, test_orientation_FF_init, (params, ))
    completeness = pool.map(test_orientation_FF_reduced, qfib.T)
    pool.close()
else:
    print("INFO:\tusing map search with paintGrid on %d processes"
          % ncpus)
    start = timeit.default_timer()

    completeness = indexer.paintGrid(
        qfib,
        eta_ome,
        etaRange=np.radians(cfg.find_orientations.eta.range),
        omeTol=np.radians(cfg.find_orientations.omega.tolerance),
        etaTol=np.radians(cfg.find_orientations.eta.tolerance),
        omePeriod=np.radians(cfg.find_orientations.omega.period),
        threshold=on_map_threshold,
        doMultiProc=ncpus > 1,
        nCPUs=ncpus
        )
    print("INFO:\t\t...took %f seconds" % (timeit.default_timer() - start))
completeness = np.array(completeness)


# %%
# =============================================================================
# CLUSTERING AND GRAINS OUTPUT
# =============================================================================
if not os.path.exists(cfg.analysis_dir):
    os.makedirs(cfg.analysis_dir)
qbar_filename = 'accepted_orientations_' + analysis_id + '.dat'

print("INFO:\trunning clustering using '%s'"
      % cfg.find_orientations.clustering.algorithm)
start = timeit.default_timer()

qbar, cl = run_cluster(
    completeness, qfib, plane_data.getQSym(), cfg,
    min_samples=min_samples,
    compl_thresh=compl_thresh,
    radius=cl_radius)

print("INFO:\t\t...took %f seconds" % (timeit.default_timer() - start))
print("INFO:\tfound %d grains; saved to file: '%s'"
      % (qbar.shape[1], qbar_filename))

np.savetxt(qbar_filename, qbar.T,
           fmt='%.18e', delimiter='\t')

gw = instrument.GrainDataWriter(os.path.join(cfg.analysis_dir, 'grains.out'))
grain_params_list = []
for gid, q in enumerate(qbar.T):
    phi = 2*np.arccos(q[0])
    n = xfcapi.unitRowVector(q[1:])
    grain_params = np.hstack([phi*n, cnst.zeros_3, cnst.identity_6x1])
    gw.dump_grain(gid, 1., 0., grain_params)
    grain_params_list.append(grain_params)
gw.close()
