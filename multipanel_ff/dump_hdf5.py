#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 18:52:32 2017

@author: s1iduser
"""
try:
    import dill as cpl
except(ImportError):
    import cPickle as cpl

import glob

import numpy as np

import os

import yaml

from hexrd import config
from hexrd import instrument
from hexrd import imageseries
from hexrd.imageseries.omega import OmegaImageSeries


# instrument
def load_instrument(yml):
    with file(yml, 'r') as f:
        icfg = yaml.load(f)
    instr = instrument.HEDMInstrument(instrument_config=icfg)
    for det_key in instr.detectors:
        if 'saturation_level' in icfg['detectors'][det_key].keys():
            sat_level = icfg['detectors'][det_key]['saturation_level']
            print("INFO: Setting panel '%s' saturation level to %e"
                  % (det_key, sat_level))
            instr.detectors[det_key].saturation_level = sat_level
    return instr


# plane data
def load_pdata(cpkl, key):
    with file(cpkl, "r") as matf:
        mat_list = cpl.load(matf)
    return dict(zip([i.name for i in mat_list], mat_list))[key].planeData


# images
def load_images(yml):
    return imageseries.open(yml, format="frame-cache", style="npz")


# %%
cfg_filename = 'Au_config.yml'
samp_name = 'Au_hydra'
scan_number = 0

data_dir = os.getcwd()

fc_stem = "%s_%06d-fc_%%s.npz" % (samp_name, scan_number)

grain_id = 1
tol_loop_idx = -1
max_tth = 17.

block_number = 0    # ONLY FOR TIMESERIES CFG

# %%
cfg = config.open(cfg_filename)[block_number]

# load instrument
instr = load_instrument(cfg.instrument.parameters)
det_keys = instr.detectors.keys()

# !!! panel buffer setting is global and assumes same typ of panel!
for det_key in det_keys:
    instr.detectors[det_key].panel_buffer = \
        np.array(cfg.fit_grains.panel_buffer)
    buff_str = str(np.array(cfg.fit_grains.panel_buffer))
    print("INFO: set panel buffer for %s to: %s" % (det_key, buff_str))

# load plane data
plane_data = load_pdata(cfg.material.definitions, cfg.material.active)
if max_tth:
    if not isinstance(max_tth, bool):
        plane_data.exclusions = np.zeros_like(plane_data.exclusions, dtype=bool)
        plane_data.tThMax = np.radians(max_tth)
        print("setting max bragg angles to %f degrees" % max_tth)

# images
imsd = dict.fromkeys(det_keys)
for det_key in det_keys:
    ims_fname = glob.glob(fc_stem % det_key)[0]
    imsd[det_key] = OmegaImageSeries(load_images(ims_fname))

# %%
tth_tol = cfg.fit_grains.tolerance.tth[tol_loop_idx]
eta_tol = cfg.fit_grains.tolerance.eta[tol_loop_idx]
ome_tol = cfg.fit_grains.tolerance.omega[tol_loop_idx]
npdiv = cfg.fit_grains.npdiv
threshold = cfg.fit_grains.threshold

eta_ranges = np.radians(cfg.find_orientations.eta.range)
ome_period = np.radians(cfg.find_orientations.omega.period)
# %%
grains_file = os.path.join(cfg.analysis_dir, 'grains.out')

grains_table = np.loadtxt(grains_file, ndmin=2)
grain = grains_table[grain_id]
grain_params = grain[3:15]

# %%
# =============================================================================
# Calls to output
# =============================================================================

complvec, results = instr.pull_spots(
    plane_data, grain_params,
    imsd,
    tth_tol=tth_tol,
    eta_tol=eta_tol,
    ome_tol=ome_tol,
    npdiv=npdiv, threshold=threshold,
    eta_ranges=eta_ranges,
    ome_period=ome_period,
    dirname=cfg.analysis_dir, filename='grain_%05d_nearest' % grain_id,
    save_spot_list=False, output_format='hdf5',
    quiet=True, check_only=False, interp='nearest')

# %%
complvec, results = instr.pull_spots(
    plane_data, grain_params,
    imsd,
    tth_tol=tth_tol,
    eta_tol=eta_tol,
    ome_tol=ome_tol,
    npdiv=npdiv, threshold=threshold,
    eta_ranges=eta_ranges,
    ome_period=ome_period,
    dirname=cfg.analysis_dir, filename='grain_%05d_bilinear' % grain_id,
    save_spot_list=False, output_format='hdf5',
    quiet=True, check_only=False, interp='bilinear')

# %%
complvec, results = instr.pull_spots(
    plane_data, grain_params,
    imsd,
    tth_tol=tth_tol,
    eta_tol=eta_tol,
    ome_tol=ome_tol,
    npdiv=npdiv, threshold=threshold,
    eta_ranges=eta_ranges,
    ome_period=ome_period,
    dirname=cfg.analysis_dir, filename='spots_%05d.out' % grain_id,
    save_spot_list=False,
    quiet=True, check_only=False, interp='nearest')
