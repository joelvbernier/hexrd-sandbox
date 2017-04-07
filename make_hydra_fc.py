import os
import yaml

import numpy as np

from hexrd import instrument
from hexrd import imageseries

Pims = imageseries.process.ProcessedImageSeries

#==============================================================================
#%% INITIALIZATION
#==============================================================================

# make input yaml files
working_dir = '/Users/Shared/APS/Hydra/Apr12'
image_dir = '/Users/Shared/APS/Hydra/Apr12'
image_stem = 'Ruby2_%05d.%s'

instr_cfg_file = os.path.join(working_dir, 'Hydra_Apr12.yml')
icfg = yaml.load(open(instr_cfg_file, 'r'))
instr = instrument.HEDMInstrument(instrument_config=icfg)

yml_tmplate = '''
image-files:
  directory: %s
  files: "%s"
options:
  empty-frames: 2
  max-file-frames: %d
meta:
  panel: %s
'''

nf = 160
delta_ome = 0.26212445576286464
wedge_width = nf*delta_ome
wedge_args = [
    (0, wedge_width, nf),
    (45, 45+wedge_width, nf),
    (90, 90+wedge_width, nf),
    (135, 135+wedge_width, nf),
]    
# FIXME: UGH, scan ranges by hand :-(
det_keys = ['GE1', 'GE2', 'GE3', 'GE4']
scan_ranges = [415, 416, 417, 418]

#==============================================================================
#%% LOOP WRITES OVER DETECTOR PANELS
#==============================================================================

for det_key in det_keys:
    image_list_str = '\n'
    for i in scan_ranges:
        image_list_str += image_stem %(i, det_key.lower()) + '\n'
    output_str = yml_tmplate % (
        image_dir, image_list_str, nf, det_key)
    rawfname = "raw_images_%s.yml"
    with open(rawfname %det_key, 'w') as f:
        print >> f, output_str

    # load basic imageseries: no flip, no omegas
    ims = imageseries.open(rawfname %det_key, 'image-files')
    if len(ims) != len(scan_ranges)*nf:
        import pdb; pdb.set_trace()

    # generate omegas
    w = imageseries.omega.OmegaWedges(len(scan_ranges)*nf)
    for wa in wedge_args:
        w.addwedge(*wa)
    meta = ims.metadata
    meta['omega'] = w.omegas
    w.save_omegas('ruby_omegas.npy')
    print ims.metadata

    # make dark
    print "making dark image"
    dark = imageseries.stats.median(ims, nframes=100)
    np.save('median_dark_%s.npy' %det_key, dark)

    # add processing opts
    pims = Pims(ims, [('dark', dark), ('flip', 'h')])

    # save as frame-cache
    print "writing frame cache"
    imageseries.write(pims,
                      'imageseries-fc_%s.yml' %(det_key),
                      'frame-cache',
                      cache_file="images-fc_%s.npz" %(det_key),
                      threshold=10)
