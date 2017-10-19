#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 19:04:10 2017

@author: bernier2
"""
import os

import multiprocessing

import numpy as np

try:
    import dill as cpl
except(ImportError):
    import cPickle as cpl

import yaml

from hexrd import config
from hexrd import imageseries
from hexrd.imageseries.omega import OmegaImageSeries
from hexrd import instrument
from hexrd.xrd.transforms_CAPI import angularDifference, mapAngle

from hexrd.xrd.fitting import fitGrain, objFuncFitGrain, \
    gFlag_ref, gScl_ref

# instrument
def load_instrument(yml):
    with file(yml, 'r') as f:
        icfg = yaml.load(f)
    return instrument.HEDMInstrument(instrument_config=icfg)

# images
def load_images(yml):
    return imageseries.open(yml, "frame-cache")

#==============================================================================
#%% USER INPUT
#==============================================================================
matl_filename = 'materials.hexrd'
matl_key = 'steel'

tth_max = np.radians(21)

instr_filename = 'dexelas_f2_Apr17.yml'

image_stem = "%s_%05d"
fc_dir_stem = image_stem + "-fcache-dir"

h5_file_number = 108

cfg_filename =  'SS-304L2.yml'

omegas_filename = 'fastsweep_omegas_360.npy'
eta_ranges = None

# FITTING PARAMS
tth_tol = 0.25
eta_tol = 2.0
ome_tol = 2.0
npdiv = 2
threshold = 400  # FIXME: should read this form frame-cache
#==============================================================================
#%% INITIALIZATION
#==============================================================================
cfg = config.open(cfg_filename)[0]

instr = load_instrument(instr_filename)
det_keys = instr.detectors.keys()

mat_list = cpl.load(open(matl_filename, 'r'))
plane_data = dict(
    zip([i.name for i in mat_list], mat_list)
)[matl_key].planeData
plane_data.tThMax = None
plane_data.exclusions = np.zeros_like(plane_data.exclusions, dtype=bool)
plane_data.tThMax = tth_max

# FIXME: should get this from imageseries directly
omegas_array = np.load(omegas_filename)

imgser_dict = dict.fromkeys(det_keys)
for det_key in det_keys:
    str_tuple = (det_key.lower(), h5_file_number)
    yml_file = os.path.join(
            fc_dir_stem % str_tuple, 
            image_stem % str_tuple + '-fcache.yml')
    ims = load_images(yml_file)
    # FIXME: imageseries needs to have omegas; kludged for now
    ims.metadata['omega'] = omegas_array
    imgser_dict[det_key] = OmegaImageSeries(ims)
    

# CAVEAT: the omega ranges in each imageseries in the dict are implied to be
# identical.  Steps are also assumed to always be positive/CCW in this context.
# These quantities are also in DEGREES <JVB 2017-03-26>
ome_period = np.radians(cfg.find_orientations.omega.period)

ncpus = multiprocessing.cpu_count()

#==============================================================================
#%% FITTING
#==============================================================================
grains_filename = os.path.join(cfg.analysis_dir, 'grains.out')
grains_table = np.loadtxt(grains_filename, ndmin=2)
spots_filename = "spots_%05d.out"
params = dict(
        grains_table=grains_table,
        plane_data=plane_data, 
        instrument=instr,
        imgser_dict=imgser_dict,
        tth_tol=tth_tol, 
        eta_tol=eta_tol,
        ome_tol=ome_tol,
        npdiv=npdiv,
        threshold=threshold, 
        eta_ranges=eta_ranges,
        ome_period=ome_period,
        analysis_dirname=cfg.analysis_dir, 
        spots_filename=spots_filename)

#==============================================================================
#%% ORIENTATION SCORING
#==============================================================================

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
    ome_period =paramMP['ome_period']
    analysis_dirname = paramMP['analysis_dirname'] 
    spots_filename = paramMP['spots_filename']
        
    grain = grains_table[grain_id]
    grain_params = grain[3:15]
        
    complvec, results = instrument.pull_spots(
        plane_data, grain_params,
        imgser_dict,
        tth_tol=tth_tol, eta_tol=eta_tol, ome_tol=ome_tol,
        npdiv=npdiv, threshold=threshold,
        eta_ranges=eta_ranges,
        ome_period=ome_period,
        dirname=analysis_dirname, filename=spots_filename %grain_id, 
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
    
    if num_refl_valid <= 12: # not enough reflections to fit... exit
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


#%%
pool = multiprocessing.Pool(ncpus, fit_grain_FF_init, (params, ))
fit_results = pool.map(
    fit_grain_FF_reduced, 
    np.array(grains_table[:, 0], dtype=int)
)
pool.close()


#%%
gw = instrument.GrainDataWriter(grains_filename)
for fit_result in fit_results:
    gw.dump_grain(*fit_result)
    pass
gw.close()
