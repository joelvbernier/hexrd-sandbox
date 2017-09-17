#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 19:04:10 2017

@author: bernier2
"""
import glob

import os

import multiprocessing

import numpy as np

import timeit

try:
    import dill as cpl
except(ImportError):
    import cPickle as cpl

import yaml

from hexrd import config
from hexrd import constants as cnst
from hexrd import imageseries
from hexrd.imageseries.omega import OmegaImageSeries
from hexrd import instrument
from hexrd.xrd import transforms_CAPI as xfcapi

from hexrd.xrd.fitting import fitGrain, objFuncFitGrain, gFlag_ref


# plane data
def load_pdata(cpkl, key):
    with file(cpkl, "r") as matf:
        mat_list = cpl.load(matf)
    return dict(zip([i.name for i in mat_list], mat_list))[key].planeData


# instrument
def load_instrument(yml):
    with file(yml, 'r') as f:
        icfg = yaml.load(f)
    return instrument.HEDMInstrument(instrument_config=icfg)


# images
def load_images(yml):
    return imageseries.open(yml, format="frame-cache", style="npz")


# =============================================================================
# %% USER INPUT
# =============================================================================

# cfg file -- currently ignores image_series block
cfg_filename = 'SS-304L2.yml'

# FIXME:  reconcile with config file!!! Goes for pretty much all the options
raw_data_dir_template = '/nfs/chess/raw/2017-1/f2/%s/%s/%d/ff/*.h5'
expt_name = 'pokharel-603-1'
samp_name = 'SS-304L2'
scan_number = 11
data_dir = os.getcwd()

clobber_grains = True

# =============================================================================
# %% INITIALIZATION
# =============================================================================
cfg = config.open(cfg_filename)[0]

analysis_id = '%s_%s' % (
    cfg.analysis_name.strip().replace(' ', '-'),
    cfg.material.active.strip().replace(' ', '-'),
    )

grains_filename = os.path.join(
    cfg.analysis_dir, 'grains.out'
)

max_tth = cfg.fit_grains.tth_max
try:
    max_tth = np.radians(float(max_tth))
except(ValueError):
    max_tth = None

# load plane data
plane_data = load_pdata(cfg.material.definitions, cfg.material.active)
plane_data.exclusions = np.zeros_like(plane_data.exclusions, dtype=bool)
plane_data.tThMax = max_tth

# load instrument
instr = load_instrument(cfg.instrument.parameters)
det_keys = instr.detectors.keys()


fc_stem = "%s_%d_%%s_*-cachefile.npz" % (samp_name, scan_number)

imsd = dict.fromkeys(det_keys)
for det_key in det_keys:
    ims_fname = glob.glob(fc_stem % det_key.lower())[0]
    imsd[det_key] = OmegaImageSeries(load_images(ims_fname))

# CAVEAT: the omega ranges in each imageseries in the dict are implied to be
# identical.  Steps are also assumed to always be positive/CCW in this context.
# These quantities are also in DEGREES <JVB 2017-03-26>
# grab eta ranges
eta_ranges = np.radians(cfg.find_orientations.eta.range)
ome_period = np.radians(cfg.find_orientations.omega.period)

ncpus = cfg.multiprocessing

# =============================================================================
# %% FITTING
# =============================================================================

grains_filename = os.path.join(cfg.analysis_dir, 'grains.out')

# make sure grains.out is there...
if not os.path.exists(grains_filename) or clobber_grains:
    qbar = np.loadtxt('accepted_orientations_' + analysis_id + '.dat').T

    gw = instrument.GrainDataWriter(grains_filename)
    grain_params_list = []
    for i_g, q in enumerate(qbar.T):
        phi = 2*np.arccos(q[0])
        n = xfcapi.unitRowVector(q[1:])
        grain_params = np.hstack([phi*n, cnst.zeros_3, cnst.identity_6x1])
        gw.dump_grain(int(i_g), 1., 0., grain_params)
        grain_params_list.append(grain_params)
    gw.close()
    pass

grains_table = np.loadtxt(grains_filename, ndmin=2)
spots_filename = "spots_%05d.out"
params = dict(
        grains_table=grains_table,
        plane_data=plane_data,
        instrument=instr,
        imgser_dict=imsd,
        tth_tol=cfg.fit_grains.tolerance.tth,
        eta_tol=cfg.fit_grains.tolerance.eta,
        ome_tol=cfg.fit_grains.tolerance.omega,
        npdiv=cfg.fit_grains.npdiv,
        threshold=cfg.fit_grains.threshold,
        eta_ranges=eta_ranges,
        ome_period=ome_period,
        analysis_dirname=cfg.analysis_dir,
        spots_filename=spots_filename)

# =============================================================================
# %% ORIENTATION SCORING
# =============================================================================


def fit_grain_FF_init(params):
    global paramMP
    paramMP = params


def fit_grain_FF_reduced(grain_id):
    """
    input parameters are [
    plane_data, instrument, imgser_dict,
    tth_tol, eta_tol, ome_tol, npdiv, threshold
    ]
    """
    grains_table = paramMP['grains_table']
    plane_data = paramMP['plane_data']
    instrument = paramMP['instrument']
    imgser_dict = paramMP['imgser_dict']
    tth_tol = paramMP['tth_tol']
    eta_tol = paramMP['eta_tol']
    ome_tol = paramMP['ome_tol']
    npdiv = paramMP['npdiv']
    threshold = paramMP['threshold']
    eta_ranges = paramMP['eta_ranges']
    ome_period = paramMP['ome_period']
    analysis_dirname = paramMP['analysis_dirname']
    spots_filename = paramMP['spots_filename']

    grain = grains_table[grain_id]
    grain_params = grain[3:15]

    complvec, results = instrument.pull_spots(
        plane_data, grain_params,
        imgser_dict,
        tth_tol=tth_tol[0], eta_tol=eta_tol[0], ome_tol=ome_tol[0],
        npdiv=npdiv, threshold=threshold,
        eta_ranges=eta_ranges,
        ome_period=ome_period,
        dirname=analysis_dirname, filename=spots_filename % grain_id,
        save_spot_list=False,
        quiet=True, lrank=1, check_only=False)

    # ======= DETERMINE VALID REFLECTIONS =======

    # CAVEAT: in the event of different stauration levels, can't mark saturated
    # spots in aggregated results <JVB 2017-03-26>
    culled_results = dict.fromkeys(results)
    num_refl_tot = 0
    num_refl_valid = 0
    for det_key in culled_results:
        presults = results[det_key]

        valid_refl_ids = np.array([x[0] for x in presults]) >= 0

        # FIXME: spot saturations will have to be handled differently
        unsat_spots = np.ones(len(valid_refl_ids))

        idx = np.logical_and(valid_refl_ids, unsat_spots)

        # TODO: wire in reflection overlap tables
        """
        # if an overlap table has been written, load it and use it
        overlaps = np.zeros(len(refl_table), dtype=bool)
        try:
            ot = np.load(self._p['overlap_table'])
            for key in ot.keys():
                for this_table in ot[key]:
                    these_overlaps = np.where(
                        this_table[:, 0] == grain_id)[0]
                    if len(these_overlaps) > 0:
                        mark_these = np.array(
                            this_table[these_overlaps, 1], dtype=int
                        )
                        overlaps[mark_these] = True
            idx = np.logical_and(idx, ~overlaps)
        except IOError, IndexError:
            #print "no overlap table found"
            pass
        """

        # attach to proper dict entry
        culled_results[det_key] = [presults[i] for i in np.where(idx)[0]]
        num_refl_tot += len(valid_refl_ids)
        num_refl_valid += sum(valid_refl_ids)
        pass

    # CAVEAT: completeness from pullspots only; incl saturated and overlaps
    # <JVB 2015-12-15>
    completeness = num_refl_valid / float(num_refl_tot)

    # ======= DO LEASTSQ FIT =======

    if num_refl_valid <= 12:    # not enough reflections to fit... exit
        grain_params_fit = grain_params
        return grain_id, completeness, np.inf, grain_params_fit
    else:
        grain_params_fit = fitGrain(
                grain_params, instrument, culled_results,
                plane_data.latVecOps['B'], plane_data.wavelength
            )
        # get chisq
        # TODO: do this while evaluating fit???
        chisq = objFuncFitGrain(
                grain_params, grain_params, gFlag_ref,
                instrument,
                culled_results,
                plane_data.latVecOps['B'], plane_data.wavelength,
                ome_period,
                simOnly=False, return_value_flag=2)

        return grain_id, completeness, chisq, grain_params_fit


# =============================================================================
# %% EXECUTE MP FIT
# =============================================================================

# DO FIT!
print("INFO:\tstarting fit on %d processes" % min(ncpus, len(grains_table)))
start = timeit.default_timer()
pool = multiprocessing.Pool(min(ncpus, len(grains_table)),
                            fit_grain_FF_init,
                            (params, ))
fit_results = pool.map(
    fit_grain_FF_reduced,
    np.array(grains_table[:, 0], dtype=int)
)
pool.close()
elapsed = timeit.default_timer() - start
print("INFO: fitting took %f seconds" % elapsed)

# =============================================================================
# %% WRITE OUTPUT
# =============================================================================

gw = instrument.GrainDataWriter(grains_filename)
for fit_result in fit_results:
    gw.dump_grain(*fit_result)
    pass
gw.close()
