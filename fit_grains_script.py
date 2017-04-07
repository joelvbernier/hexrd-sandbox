#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 19:04:10 2017

@author: bernier2
"""
import sys
import os

import numpy as np

import dill

import yaml

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

#==============================================================================
#%% USER INPUT
#==============================================================================
matl_filename = 'materials.cpl'
matl_key = 'ruby'

instr_filename = 'Hydra_Apr12.yml'

imgser_dir = './'
imgser_stem = 'imageseries-fc_%s.yml'

grains_filename = 'results/grains.out'
##==============================================================================
#%% INITIALIZATION
#==============================================================================

instr = load_instrument(instr_filename)
det_keys = instr.detectors.keys()

mat_list = dill.load(open(matl_filename, 'r'))
plane_data = dict(
    zip([i.name for i in mat_list], mat_list)
)[matl_key].planeData

imgser_dict = dict.fromkeys(det_keys)
for det_key in det_keys:
    ims = imageseries.open(
        os.path.join(imgser_dir, imgser_stem %(det_key.upper())
        ), 'frame-cache'
    )
    imgser_dict[det_key] = OmegaImageSeries(ims)
    

# CAVEAT: the omega ranges in each imageseries in the dict are implied to be
# identical.  Steps are also assumed to always be positive/CCW in this context.
# These quantities are also in DEGREES <JVB 2017-03-26>
ome_start = imgser_dict[det_key].omega[0, 0]
ome_stop = imgser_dict[det_key].omega[-1, 1]
ome_step = imgser_dict[det_key].omega[0, 1] - imgser_dict[det_key].omega[0, 0]
ome_period = np.radians([ome_start, ome_start + 360])

# FIXME: this test will fail if omega spec wraps around 0, e.g.
# [(0, 60), (-60, 0)] --> yields nwedges = 2 <JVB 2017-03-26>
full_range = np.logical_and(
    angularDifference(ome_start, ome_stop, units='degrees')[0] < 1e-3, 
    imgser_dict[det_key].nwedges
)
#==============================================================================
#%% FITTING
#==============================================================================
grains_table = np.loadtxt(grains_filename, ndmin=2)
gw = instrument.GrainDataWriter(grains_filename)
grain_params_fit = []
for grain in grains_table:
    gid = int(grain[0])
    grain_params = grain[3:15]
    spots_filename = "spots_%05d.out" %gid
    
    complvec, results = instr.pull_spots(
        plane_data, grain_params,
        imgser_dict,
        tth_tol=0.25, eta_tol=2., ome_tol=2.,
        npdiv=2, threshold=0,
        eta_ranges=[np.radians([-95, 85]), np.radians([95, 265])],
        ome_period=(0., 2.*np.pi),
        dirname='results', filename=spots_filename, save_spot_list=False,
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
        
        pred_ome = mapAngle(
            np.array([x[5][-1] for x in presults]), 
            ome_period)
            
        if not full_range:
            # if here, incomplete have omega range and
            # clip the refelctions very close to the edges to avoid
            # problems with the least squares...
            if np.sign(ome_step) < 0:
                idx_ome = np.logical_and(
                    pred_ome < np.radians(ome_start + 2*ome_step),
                    pred_ome > np.radians(ome_stop - 2*ome_step)
                    )
            else:
                idx_ome = np.logical_and(
                    pred_ome > np.radians(ome_start + 2*ome_step),
                    pred_ome < np.radians(ome_stop - 2*ome_step)
                    )
            idx = np.logical_and(
                valid_refl_ids,
                np.logical_and(unsat_spots, idx_ome)
                )
        else:
            idx = np.logical_and(valid_refl_ids, unsat_spots)
            pass  # end if edge case

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
        grain_params[1] = np.inf
        grain_params_fit.append(grain_params)
    else:
        grain_params_fit.append(
            fitGrain(
                    grain_params, instr, culled_results,
                    plane_data.latVecOps['B'], plane_data.wavelength
            )
        )
    # get chisq
    # TODO: do this while evaluating fit???
    chisq = objFuncFitGrain(
        grain_params, grain_params, gFlag_ref,
        instr,
        culled_results,
        plane_data.latVecOps['B'], plane_data.wavelength,
        ome_period,
        simOnly=False, return_value_flag=2)
    gw.dump_grain(
        gid, sum(complvec)/float(len(complvec)), 
        chisq, grain_params_fit[-1])
    pass
gw.close()
