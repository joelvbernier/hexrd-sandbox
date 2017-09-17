#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 19:04:10 2017

@author: bernier2
"""
from __future__ import print_function

import os

import multiprocessing

import numpy as np

from scipy import ndimage

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

from matplotlib import pyplot as plt

# plane data
def load_pdata(cpkl, key):
    with file(cpkl, "r") as matf:
        mat_list = cpl.load(matf)
    return dict(zip([i.name for i in mat_list], mat_list))[mat_key].planeData

# images
def load_images(yml):
    return imageseries.open(yml, "frame-cache")

# instrument
def load_instrument(yml):
    with file(yml, 'r') as f:
        icfg = yaml.load(f)
    return instrument.HEDMInstrument(instrument_config=icfg)

#%%
#==============================================================================
# START USER INPUT
#==============================================================================

# FIXME:  reconcile with config file!!! Goes for pretty much all the options

materials_fname = 'materials.hexrd'
mat_key = 'steel'

instr_config_filename = 'dexelas_f2_Apr17.yml'

cfg_filename = 'SS-304L2.yml'

image_stem = "%s_%05d"
fc_dir_stem = image_stem + "-fcache-dir"

h5_file_number = 108

# for indexing
map_threshold = 1000
fiber_gen_threshold = 1000
fiber_ndiv = 720
fiber_seeds = [0,1]
tth_tol=0.2
eta_tol=1.0
ome_tol=1.0
npdiv=1

min_samples=5
compl_thresh=0.75
cl_radius=1.0

omegas_filename = 'fastsweep_omegas_360.npy'

max_tth = np.radians(12.5)

make_max_frames = True
max_frames_output_name = 'SS-304L2_cal_scan_001.hdf5'
#==============================================================================
# END USER INPUT
#==============================================================================

#%%
cfg = config.open(cfg_filename)[0]

analysis_id = '%s_%s' %(
    cfg.analysis_name.strip().replace(' ', '-'),
    cfg.material.active.strip().replace(' ', '-'),
    )

# load plane data
pd = load_pdata(materials_fname, mat_key)
pd.tThMax = max_tth

# load instrument
instr = load_instrument(instr_config_filename)
det_keys = instr.detectors.keys()

# load omegas
# WARNING: using master array in root directory!
# FIXME: change ASAP to make saved caches contain omega info!!!
omegas_array = np.load(omegas_filename)

#%%

imsd = dict.fromkeys(det_keys)
for det_key in det_keys:
    str_tuple = (det_key.lower(), h5_file_number)
    yml_file = os.path.join(
            fc_dir_stem % str_tuple, 
            image_stem % str_tuple + '-fcache.yml')
    ims = load_images(yml_file)
    # FIXME: imageseries needs to have omegas; kludged for now
    ims.metadata['omega'] = omegas_array
    imsd[det_key] = OmegaImageSeries(ims)


if make_max_frames:
    max_frames = dict.fromkeys(det_keys)
    for det_key in det_keys:
        max_frames[det_key] = np.max(
                np.array([i for i in imsd[det_key]]), 
                axis=0
            )
    ims_out = imageseries.open(
            None, 'array', 
            data=np.array([max_frames[i] for i in max_frames]), 
            meta={'panels':max_frames.keys()}
        )
    imageseries.write(
            ims_out, max_frames_output_name, 
            'hdf5', path='/imageseries'
        )
#%%

# note that here maps are only used for quaternion generation
active_hkls = fiber_seeds

# omega period... 
# QUESTION: necessary???
ome_period = np.radians(cfg.find_orientations.omega.period)

# make eta_ome maps
eta_ome = instrument.GenerateEtaOmeMaps(
    imsd, instr, pd, 
    active_hkls=active_hkls, threshold=map_threshold)

#%%
cmap = plt.cm.hot
cmap.set_under('b')

fig, ax = plt.subplots()
i_ring = 0
this_map_f = -ndimage.filters.gaussian_laplace(
    eta_ome.dataStore[i_ring], 1.0,
)
ax.imshow(this_map_f, interpolation='nearest', 
          vmin=0.1, vmax=1.0, cmap=cmap)
labels, num_spots = ndimage.label(
    this_map_f > 0, ndimage.generate_binary_structure(2, 1)
)
coms = np.atleast_2d(
    ndimage.center_of_mass(
        eta_ome.dataStore[i_ring], labels=labels, index=range(1, num_spots+1)
    )
)
ax.hold(True)
ax.plot(coms[:, 1], coms[:, 0], 'm+', ms=12)
ax.axis('tight')
#%%
#==============================================================================
# SEARCH SPACE GENERATION
#==============================================================================
ncpus = multiprocessing.cpu_count()  # USE ALL BY DEFAULT!!!
qfib = generate_orientation_fibers(
    eta_ome, instr.chi, fiber_gen_threshold, 
    fiber_seeds, fiber_ndiv, 
    ncpus=ncpus)
print("INFO: will test %d quaternions" %qfib.shape[1])
#%%
#==============================================================================
# ORIENTATION SCORING
#==============================================================================

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
        plane_data=pd, 
        instrument=instr,
        imgser_dict=imsd,
        tth_tol=tth_tol, 
        eta_tol=eta_tol,
        ome_tol=ome_tol,
        npdiv=npdiv,
        threshold=fiber_gen_threshold)

pool = multiprocessing.Pool(ncpus, test_orientation_FF_init, (params, ))
completeness = pool.map(test_orientation_FF_reduced, qfib.T)
pool.close()


#%%
#==============================================================================
# CLUSTERING AND GRAINS OUTPUT
#==============================================================================
if not os.path.exists(cfg.analysis_dir):
    os.makedirs(cfg.analysis_dir)

qbar, cl = run_cluster(completeness, qfib, pd.getQSym(), cfg, 
            min_samples=min_samples, 
            compl_thresh=compl_thresh, 
            radius=cl_radius)

np.savetxt('accepted_orientations_' + analysis_id + '.dat', qbar.T, 
           fmt='%.18e', delimiter='\t')

gw = instrument.GrainDataWriter(os.path.join(cfg.analysis_dir, 'grains.out'))
grain_params_list = []
for i_g, q in enumerate(qbar.T):
    phi = 2*np.arccos(q[0])
    n = xfcapi.unitRowVector(q[1:])
    grain_params = np.hstack([phi*n, cnst.zeros_3, cnst.identity_6x1])
    gw.dump_grain(int(i_g), 1., 0., grain_params)
    grain_params_list.append(grain_params)
gw.close()
