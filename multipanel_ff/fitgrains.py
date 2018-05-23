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
    instr = instrument.HEDMInstrument(instrument_config=icfg)
    for det_key in instr.detectors:
        if 'saturation_level' in icfg['detectors'][det_key].keys():
            sat_level = icfg['detectors'][det_key]['saturation_level']
            print("INFO: Setting panel '%s' saturation level to %e"
                  % (det_key, sat_level))
            instr.detectors[det_key].saturation_level = sat_level
    return instr


# images
def load_images(yml):
    return imageseries.open(yml, format="frame-cache", style="npz")


# =============================================================================
# %% USER INPUT
# =============================================================================

cfg_filename = 'Au_config.yml'
samp_name = 'Au_hydra'
scan_number = 0

data_dir = os.getcwd()

fc_stem = "%s_%06d-fc_%%s.npz" % (samp_name, scan_number)

block_number = 0    # ONLY FOR TIMESERIES CFG

clobber_grains = False

tol_loop_idx = -1

threshold = None

# =============================================================================
# %% INITIALIZATION
# =============================================================================
cfg = config.open(cfg_filename)[block_number]

analysis_id = '%s_%s' % (
    cfg.analysis_name.strip().replace(' ', '-'),
    cfg.material.active.strip().replace(' ', '-'),
    )

grains_filename = os.path.join(
    cfg.analysis_dir, 'grains.out'
)

# !!! handle max_tth config option
max_tth = cfg.fit_grains.tth_max
if max_tth:
    if type(cfg.fit_grains.tth_max) != bool:
        max_tth = np.radians(float(max_tth))
else:
    max_tth = None

# load plane data
plane_data = load_pdata(cfg.material.definitions, cfg.material.active)
if max_tth is not None:
    plane_data.exclusions = np.zeros_like(plane_data.exclusions, dtype=bool)
    plane_data.tThMax = max_tth

# load instrument
instr = load_instrument(cfg.instrument.parameters)
det_keys = instr.detectors.keys()

# !!! panel buffer setting is global and assumes same typ of panel!
for det_key in det_keys:
    instr.detectors[det_key].panel_buffer = \
        np.array(cfg.fit_grains.panel_buffer)
    buff_str = str(np.array(cfg.fit_grains.panel_buffer))
    print("INFO: set panel buffer for %s to: %s" % (det_key, buff_str))

imsd = dict.fromkeys(det_keys)
for det_key in det_keys:
    ims_fname = glob.glob(fc_stem % det_key)[0]
    imsd[det_key] = OmegaImageSeries(load_images(ims_fname))

# CAVEAT: the omega ranges in each imageseries in the dict are implied to be
# identical.  Steps are also assumed to always be positive/CCW in this context.
# These quantities are also in DEGREES <JVB 2017-03-26>
# grab eta ranges
eta_ranges = np.radians(cfg.find_orientations.eta.range)
ome_period = np.radians(cfg.find_orientations.omega.period)

ncpus = cfg.multiprocessing

if threshold is None:
    threshold = cfg.fit_grains.threshold

# =============================================================================
# %% FITTING
# =============================================================================

# make sure grains.out is there...
if not os.path.exists(grains_filename) or clobber_grains:
    try:
        qbar = np.loadtxt('accepted_orientations_' + analysis_id + '.dat',
                          ndmin=2).T

        gw = instrument.GrainDataWriter(grains_filename)
        grain_params_list = []
        for i_g, q in enumerate(qbar.T):
            phi = 2*np.arccos(q[0])
            n = xfcapi.unitRowVector(q[1:])
            grain_params = np.hstack([phi*n, cnst.zeros_3, cnst.identity_6x1])
            gw.dump_grain(int(i_g), 1., 0., grain_params)
            grain_params_list.append(grain_params)
        gw.close()
    except(IOError):
        if os.path.exists(cfg.fit_grains.estimate):
            grains_filename = cfg.fit_grains.estimate
        else:
            raise(RuntimeError, "neither estimate nor %s exist!"
                  % 'accepted_orientations_' + analysis_id + '.dat')
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
        refit=cfg.fit_grains.refit,
        threshold=threshold,
        eta_ranges=eta_ranges,
        ome_period=ome_period,
        analysis_dirname=cfg.analysis_dir,
        spots_filename=spots_filename)

# =============================================================================
# %% GRAIN FITTING
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
    refit = paramMP['refit']
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
        tth_tol=tth_tol[tol_loop_idx],
        eta_tol=eta_tol[tol_loop_idx],
        ome_tol=ome_tol[tol_loop_idx],
        npdiv=npdiv, threshold=threshold,
        eta_ranges=eta_ranges,
        ome_period=ome_period,
        dirname=analysis_dirname, filename=spots_filename % grain_id,
        save_spot_list=False,
        quiet=True, check_only=False, interp='nearest')

    # ======= DETERMINE VALID REFLECTIONS =======

    culled_results = dict.fromkeys(results)
    num_refl_tot = 0
    num_refl_valid = 0
    for det_key in culled_results:
        panel = instrument.detectors[det_key]

        presults = results[det_key]

        valid_refl_ids = np.array([x[0] for x in presults]) >= 0

        spot_ids = np.array([x[0] for x in presults])

        # find unsaturated spots on this panel
        if panel.saturation_level is None:
            unsat_spots = np.ones(len(valid_refl_ids))
        else:
            unsat_spots = \
                np.array([x[4] for x in presults]) < panel.saturation_level

        idx = np.logical_and(valid_refl_ids, unsat_spots)

        # if an overlap table has been written, load it and use it
        overlaps = np.zeros_like(idx, dtype=bool)
        try:
            ot = np.load(
                os.path.join(
                    analysis_dirname, os.path.join(
                        det_key, 'overlap_table.npz'
                    )
                )
            )
            for key in ot.keys():
                for this_table in ot[key]:
                    these_overlaps = np.where(
                        this_table[:, 0] == grain_id)[0]
                    if len(these_overlaps) > 0:
                        mark_these = np.array(
                            this_table[these_overlaps, 1], dtype=int
                        )
                        otidx = [
                            np.where(spot_ids == mt)[0] for mt in mark_these
                        ]
                        overlaps[otidx] = True
            idx = np.logical_and(idx, ~overlaps)
            # print("found overlap table for '%s'" % det_key)
        except(IOError, IndexError):
            # print("no overlap table found for '%s'" % det_key)
            pass

        # attach to proper dict entry
        culled_results[det_key] = [presults[i] for i in np.where(idx)[0]]
        num_refl_tot += len(valid_refl_ids)
        num_refl_valid += sum(valid_refl_ids)

        pass  # now we have culled data

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
                grain_params_fit[gFlag_ref], grain_params_fit, gFlag_ref,
                instrument,
                culled_results,
                plane_data.latVecOps['B'], plane_data.wavelength,
                ome_period,
                simOnly=False, return_value_flag=2)

        if refit is not None:
            # first get calculated x, y, ome from previous solution
            # NOTE: this result is a dict
            xyo_det_fit_dict = objFuncFitGrain(
                grain_params_fit[gFlag_ref], grain_params_fit, gFlag_ref,
                instrument,
                culled_results,
                plane_data.latVecOps['B'], plane_data.wavelength,
                ome_period,
                simOnly=True, return_value_flag=2)

            # make dict to contain new culled results
            culled_results_r = dict.fromkeys(culled_results)
            num_refl_valid = 0
            for det_key in culled_results_r:
                presults = culled_results[det_key]

                ims = imgser_dict[det_key]
                ome_step = sum(np.r_[-1, 1]*ims.metadata['omega'][0, :])

                xyo_det = np.atleast_2d(
                    np.vstack([np.r_[x[7], x[6][-1]] for x in presults])
                )

                xyo_det_fit = xyo_det_fit_dict[det_key]

                xpix_tol = refit[0]*panel.pixel_size_col
                ypix_tol = refit[0]*panel.pixel_size_row
                fome_tol = refit[1]*ome_step

                # define difference vectors for spot fits
                x_diff = abs(xyo_det[:, 0] - xyo_det_fit['calc_xy'][:, 0])
                y_diff = abs(xyo_det[:, 1] - xyo_det_fit['calc_xy'][:, 1])
                ome_diff = np.degrees(
                    xfcapi.angularDifference(xyo_det[:, 2],
                                             xyo_det_fit['calc_omes'])
                    )

                # filter out reflections with centroids more than
                # a pixel and delta omega away from predicted value
                idx_new = np.logical_and(
                    x_diff <= xpix_tol,
                    np.logical_and(y_diff <= ypix_tol,
                                   ome_diff <= fome_tol)
                                   )

                # attach to proper dict entry
                culled_results_r[det_key] = [
                    presults[i] for i in np.where(idx_new)[0]
                ]

                num_refl_valid += sum(idx_new)
                pass

            # only execute fit if left with enough reflections
            if num_refl_valid > 24:
                grain_params_fit = fitGrain(
                    grain_params_fit, instrument, culled_results_r,
                    plane_data.latVecOps['B'], plane_data.wavelength
                )
                # get chisq
                # TODO: do this while evaluating fit???
                chisq = objFuncFitGrain(
                        grain_params_fit[gFlag_ref],
                        grain_params_fit, gFlag_ref,
                        instrument,
                        culled_results_r,
                        plane_data.latVecOps['B'], plane_data.wavelength,
                        ome_period,
                        simOnly=False, return_value_flag=2)
                pass
            pass  # close refit conditional

        return grain_id, completeness, chisq, grain_params_fit


# =============================================================================
# %% EXECUTE MP FIT
# =============================================================================

# DO FIT!
if len(grains_table) == 1 or ncpus == 1:
    print("INFO:\tstarting serial fit")
    start = timeit.default_timer()
    fit_grain_FF_init(params)
    fit_results = map(
        fit_grain_FF_reduced,
        np.array(grains_table[:, 0], dtype=int)
    )
    elapsed = timeit.default_timer() - start
else:
    print("INFO:\tstarting fit on %d processes"
          % min(ncpus, len(grains_table)))
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

gw = instrument.GrainDataWriter(os.path.join(cfg.analysis_dir, 'grains.out'))
for fit_result in fit_results:
    gw.dump_grain(*fit_result)
    pass
gw.close()
