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
working_dir = '/home/beams/S1IDUSER/workspace/PUP_AFRL_Mar17'
image_dir = '/home/beams/S1IDUSER/mnt/s1b/PUP_AFRL_mar17/nf/Au2_NF'
image_stem = 'Au2_NF_%06d.tif'

instr_cfg_file = os.path.join(working_dir, 'Retiga_Mar17.yml')
icfg = yaml.load(open(instr_cfg_file, 'r'))
instr = instrument.HEDMInstrument(instrument_config=icfg)

yml_tmplate = '''
image-files:
  directory: %s
  files: "%s"

options:
  empty-frames: 0
  max-frames: 0
meta:
  panel: %s
'''

nf = 720
l0_start = 2543
wedge_args = (-90, 90, nf)
# FIXME: UGH, scan ranges by hand :-(
scan_ranges = [
        range(l0_start, l0_start+nf),
        range(l0_start+nf, l0_start+2*nf),
        range(l0_start+2*nf, l0_start+3*nf),
        range(l0_start+3*nf, l0_start+4*nf),
        ]
det_keys = ['L0', 'L1', 'L2', 'L3']
fn_dict = dict(zip(det_keys, scan_ranges))

#==============================================================================
#%% LOOP WRITES OVER DETECTOR PANELS
#==============================================================================

for det_key in det_keys:
    image_list_str = ''
    for i in fn_dict[det_key]:
        image_list_str += image_stem %i + '\n'
    output_str = yml_tmplate % (
        image_dir, image_list_str, det_key)
    rawfname = "raw_images_%s.yml"
    with open(rawfname %det_key, 'w') as f:
        print >> f, output_str

    # load basic imageseries: no flip, no omegas
    ims = imageseries.open(rawfname %det_key, 'image-files')
    if len(ims) != nf:
        import pbd; pdb.set_trace()

    # generate omegas
    w = imageseries.omega.OmegaWedges(nf)
    w.addwedge(*wedge_args)
    meta = ims.metadata
    meta['omega'] = w.omegas
    w.save_omegas('omegas_NF.npy')
    print ims.metadata

    # make dark
    print "making dark image"
    dark = imageseries.stats.median(ims, nframes=120)
    np.save('median_dark_%s.npy' %det_key, dark)

    # add processing opts
    pims = Pims(ims, [('dark', dark),])

    # save as frame-cache
    print "writing frame cache"
    imageseries.write(pims,
                      'imageseries-fc_%s.yml' %det_key,
                      'frame-cache',
                      cache_file="images-fc_%s.npz" %det_key,
                      threshold=15)
