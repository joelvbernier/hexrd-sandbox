#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 00:28:44 2017

@author: s1iduser
"""

from __future__ import print_function

import os
import yaml

import numpy as np

from hexrd import instrument
from hexrd import imageseries

import fabio

Pims = imageseries.process.ProcessedImageSeries

# =============================================================================
# INITIALIZATION
# =============================================================================

# ########################################################
# make input yaml files
working_dir = './'
raw_data_dir = '/home/joelvbernier/Working/RUBY_STANDARD/data'
instr_file = 'ge.yml'

filename_stem = 'RUBY_%04d.%%s'

output_scan_number = 0
n_wedges = 2
first_scan = 4537    # for ruby
empty_frames = 2
max_frames = 0
dark_file = None

start_ome = -60.
stop_ome = 60.

threshold = 15

popts = [('flip', 'v'), ]

instr_cfg_file = os.path.join(working_dir, instr_file)
#
# ########################################################

output_stem = filename_stem.split('.')[0] % (output_scan_number)

scan_numbers = range(first_scan, first_scan + n_wedges)
filenames_str = '\n'.join([filename_stem % scan for scan in scan_numbers])

icfg = yaml.load(open(instr_cfg_file, 'r'))
instr = instrument.HEDMInstrument(instrument_config=icfg)

raw_img_tmplate = '''
image-files:
  directory: %%s
  files: "%s"

options:
  empty-frames: %%d
  max-frames: %%d
meta:
  panel: %%s
''' % filenames_str

# =============================================================================
# LOOP WRITES OVER DETECTOR PANELS
# =============================================================================

for det_id in instr.detectors:
    suffix = det_id.lower()
    fill_tmpl = [raw_data_dir] \
        + [suffix for i in range(n_wedges)] \
        + [empty_frames, max_frames, det_id]
    # make yml string
    output_str = raw_img_tmplate % tuple(fill_tmpl)
    rawfname = "raw_images_%s-%s.yml"
    with open(rawfname % (output_stem, det_id), 'w') as f:
        print(output_str, file=f)

    # load basic imageseries: no flip, no omegas
    ims = imageseries.open(
        rawfname % (output_stem, det_id),
        'image-files')

    # generate omegas
    nf = len(ims)
    w = imageseries.omega.OmegaWedges(nf)
    w.addwedge(start_ome, stop_ome, nf)
    meta = ims.metadata
    meta['omega'] = w.omegas
    w.save_omegas('omegas_FF.npy')
    print(ims.metadata)

    # handle dark
    if dark_file is None:
        print("making dark image")
        dark = imageseries.stats.median(ims, nframes=120)
        np.save('background_%s-%s.npy' % (output_stem, det_id), dark)
    else:
        dark = fabio.open(dark_file).data
    
    # add flips
    pims = Pims(ims, [('dark', dark), ] + popts)

    # save as frame-cache
    print("writing frame cache")
    imageseries.write(pims, '%s-fc_%s.yml' % (output_stem, det_id),
                      'frame-cache',
                      cache_file="%s-fc_%s.npz" % (output_stem, det_id),
                      threshold=threshold,
                      output_yaml=False)
