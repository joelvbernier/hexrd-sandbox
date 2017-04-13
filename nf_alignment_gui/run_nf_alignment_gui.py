import sys

import glob

import cPickle
import yaml
import numpy as np

from hexrd import imageseries
from hexrd import instrument

from nf_alignment_tool import InstrumentViewer as IView

# plane data
def load_pdata(cpkl, key):
    with file(cpkl, "r") as matf:
        matlist = cPickle.load(matf)
    return dict(zip([i.name for i in matlist], matlist))[key].planeData

# images
def load_images(yml):
    return imageseries.open(yml, "image-files")

# instrument
def load_instrument(yml):
    with file(yml, 'r') as f:
        icfg = yaml.load(f)
    return instrument.HEDMInstrument(instrument_config=icfg)

if __name__ == '__main__':
    #
    #  Run viewer
    #
    print sys.argv

    # HARD CODED OMEGA RANGE
    ome_range = np.radians([-90, 90])

    instr = load_instrument(sys.argv[1])
    ims_name = sys.argv[2]
    pdata = load_pdata(sys.argv[3], sys.argv[4])
    gtable = np.loadtxt(sys.argv[5], ndmin=2)

    grain_ids_arg = sys.argv[6]
    if grain_ids_arg.lower() in ['none', '']:
        grain_ids = None
    else:
        grain_ids = np.array(grain_ids_arg.split(','), dtype=int)

    make_images = sys.argv[7]
    if sys.argv[7].lower() in ['true', '0']:
        make_images = True
        if ',' in ims_name:
            fnames = ims_name.split(',')
        else:
            fnames = glob.glob(ims_name)
        print "found the folowing files"
        ims_in = {}
        for fname in fnames:
            print "\t%s" %fname
            ims = imageseries.open(fname, 'frame-cache')
            m = ims.metadata
            ims_in[m['panel']] = ims
    else:
        print "loading single multipanel imageseries %s" % ims_name
        make_images = False
        ims_in = load_images(ims_name)

    gplist = np.atleast_2d(gtable[:, 3:15])
    iv = IView(instr, ims_in, pdata, gplist,
               grain_ids=grain_ids, make_images=make_images,
               ome_tol=0.5, ome_ranges=[ome_range,])
