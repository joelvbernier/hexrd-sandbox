#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 19:04:10 2017

@author: bernier2
"""
import os, sys

import numpy as np

try:
    import dill as cpl
except(ImportError):
    import cPickle as cpl

import yaml

from hexrd import imageseries
from hexrd import instrument
from calibrate import InstrumentViewer as IView

# plane data
def load_pdata(cpkl):
    with file(cpkl, "r") as matf:
        matlist = cpl.load(matf)
    return matlist[0].planeData

# images
def load_images(yml):
    return imageseries.open(yml, "image-files")

# instrument
def load_instrument(yml):
    with file(yml, 'r') as f:
        icfg = yaml.load(f)
    return instrument.HEDMInstrument(instrument_config=icfg)


#%%
'''
ims = load_images('./ff_max_images.yml')

Pimgs = imageseries.process.ProcessedImageSeries

mat_list = cpl.load(open('materials.cpl', 'r'))
mat_key = 'iron 2GPa'
pd = dict(zip([i.name for i in mat_list], mat_list))[mat_key].planeData

m = ims.metadata
pids = m['panels']
d = dict(zip(pids, range(len(pids))))

if 'process' in m:
    pspec = m['process']
    ops = []
    for p in pspec:
        k = p.keys()[0]
        ops.append((k, p[k]))
    pims = Pimgs(ims, ops)
else:
    pims = ims
'''
#%%
mat_list = cpl.load(open('materials.cpl', 'r'))
mat_key = 'iron 2GPa'
pd = dict(zip([i.name for i in mat_list], mat_list))[mat_key].planeData

icfg = yaml.load(open('Hydra_Apr12.yml', 'r'))
instr = instrument.HEDMInstrument(instrument_config=icfg)
det_keys = instr.detectors.keys()

'''
imgd = dict.fromkeys(instr.detectors.keys())
for pid in imgd:
    imgd[pid] = pims[d[pid]]

imgd = dict.fromkeys(det_keys)
for det_key in imgd:
    ims = imageseries.open('image_data/imageseries-fc_%s.yml' % (det_key), 'frame-cache')
    imgd[det_key] = np.max(np.array([ims[k] for k in range(len(ims))]), axis=0)
'''    
#%%
'''
pgen = generate_tth_eta(pd, instr,
                 eta_min=0., eta_max=360.,
                 pixel_size=(0.001, 0.25))
wimg = pgen.warp_image(imgd)
'''
#%%

# WARNING: assumes that all image series are the same length
pd.tThMax = np.radians(8)
n_rings = len(pd.getTTh())
imsd = dict.fromkeys(instr.detectors.keys())
for det_key in instr.detectors:
    ims = imageseries.open('image_data/imageseries-fc_%s.yml' % (det_key), 'frame-cache')
    imsd[det_key] = ims
#rp = instr.extract_line_positions(pd, imsd, collapse_tth=True, do_interpolation=False, tth_tol=0.15, eta_tol=0.25, npdiv=2)
eta_mapping = instr.extract_polar_maps(pd, imsd)

data_store = []
for i_ring in range(n_rings):
    full_map = np.zeros_like(eta_mapping[det_key][i_ring])
    nan_mask_full = np.zeros((len(eta_mapping), full_map.shape[0], full_map.shape[1]))
    i_p = 0
    for det_key, eta_map in eta_mapping.iteritems():
        nan_mask = ~np.isnan(eta_map[i_ring])
        nan_mask_full[i_p] = nan_mask
        full_map[nan_mask] = full_map[nan_mask] + eta_map[i_ring][nan_mask]
        i_p += 1
    re_nan_these = np.sum(nan_mask_full, axis=0) == 0
    full_map[re_nan_these] = np.nan
    data_store.append(full_map)

fig, ax = plt.subplots()
ax.imshow(data_store[2], interpolation='nearest')
labels, num_spots = ndimage.label(data_store[2]>1000, ndimage.generate_binary_structure(2, 1))
coms = np.atleast_2d(ndimage.center_of_mass(data_store[2], labels=labels, index=range(1, num_spots+1)))
ax.hold(True)
ax.plot(coms[:, 1], coms[:, 0], 'm+')
