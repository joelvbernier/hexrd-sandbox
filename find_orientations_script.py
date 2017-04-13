#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 19:04:10 2017

@author: bernier2
"""
import multiprocessing

import numpy as np

from scipy import ndimage

try:
    import dill as cpl
except(ImportError):
    import cPickle as cpl

import yaml

import hexrd.constants as cnst
from hexrd import imageseries
from hexrd.imageseries.omega import OmegaImageSeries
from hexrd import instrument
from hexrd import matrixutil as mutil
from hexrd.findorientations import \
generate_orientation_fibers, \
run_cluster
from hexrd.xrd import transforms_CAPI as xfcapi

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
cmap = plt.cm.hot
cmap.set_under('b')

pd.tThMax = np.radians(8)
n_rings = len(pd.getTTh())
imsd = dict.fromkeys(instr.detectors.keys())
for det_key in instr.detectors:
    ims = imageseries.open('image_data/imageseries-fc_%s.yml' % (det_key), 'frame-cache')
    imsd[det_key] = OmegaImageSeries(ims)

#rp = instr.extract_line_positions(pd, imsd, collapse_tth=True, do_interpolation=False, tth_tol=0.15, eta_tol=0.25, npdiv=2)
eta_ome = instrument.GenerateEtaOmeMaps(imsd, instr, pd, threshold=50)

fig, ax = plt.subplots()
i_ring = 0
this_map_f = -ndimage.filters.gaussian_laplace(
    eta_ome.dataStore[i_ring], 1.0,
)
ax.imshow(this_map_f, interpolation='nearest', 
          vmin=0.1, vmax=1.0, cmap=cmap)
labels, num_spots = ndimage.label(
    this_map_f > 10, ndimage.generate_binary_structure(2, 1)
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
ncpus = multiprocessing.cpu_count()
qfib = generate_orientation_fibers(eta_ome, instr.chi, 100, [0, 1, ], 720, 
                                   ncpus=ncpus)
#%%
# (quat; plane_data, instrument, imgser_dict, tth_tol, eta_tol, ome_tol, npdiv, threshold)
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
    
    compl, scrap = instr.pull_spots(
        pd, grain_params, imsd,
        tth_tol=tth_tol, eta_tol=eta_tol, ome_tol=ome_tol,
        npdiv=npdiv, threshold=threshold,
        eta_ranges=None, ome_period=(-np.pi, np.pi),
        check_only=True)

    return sum(compl)/float(len(compl))

params = dict(
        plane_data=pd, 
        instrument=instr,
        imgser_dict=imsd,
        tth_tol=0.25, 
        eta_tol=1.0,
        ome_tol=1.0,
        npdiv=1,
        threshold=50)

pool = multiprocessing.Pool(ncpus, test_orientation_FF_init, (params, ))
completeness = pool.map(test_orientation_FF_reduced, qfib.T)
pool.close()

#%%
qbar, cl = run_cluster(completeness, qfib, pd.getQSym(), cfg, 
            min_samples=5, compl_thresh=0.9, radius=1.0)

gw = instrument.GrainDataWriter('./results/grains.out')
grain_params_list = []
for q in qbar.T:
    phi = 2*np.arccos(q[0])
    n = xfcapi.unitRowVector(q[1:])
    grain_params = np.hstack([phi*n, cnst.zeros_3, cnst.identity_6x1])
    gw.dump_grain(0, 1., 0., grain_params)
    grain_params_list.append(grain_params)
gw.close()
