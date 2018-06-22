import argparse
import glob

import cPickle
import yaml
import numpy as np

from hexrd import imageseries
from hexrd import instrument

from nf_alignment_tool import InstrumentViewer as IView

import pdb

# plane data
def load_pdata(cpkl, key, tth_max=None):
    with file(cpkl, "r") as matf:
        matlist = cPickle.load(matf)
        pd = dict(zip([i.name for i in matlist], matlist))[key].planeData
        if tth_max is not None:
            pd.exclusions = np.zeros_like(pd.exclusions, detup=bool)
            pd.tThMax = np.radians(tth_max)
    return pd


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
    parser = argparse.ArgumentParser(
        description="plot grain sim over an interactive, renedered multipanel imageseries"
    )

    parser.add_argument('instrument_cfg', help="instrument config YAML file", type=str)
    parser.add_argument('imageseries_file', help="multipanel imageseries file", type=str)
    parser.add_argument('materials_file', help="materials database", type=str)
    parser.add_argument('materials_key', help="key for material in database", type=str)
    parser.add_argument('grains_file', help="far-field grains.out for simulation", type=str)
    parser.add_argument('grain_ids', help="grain id in table", type=str)

    parser.add_argument('-d', '--slider-delta', help="+/- delta for slider range", type=float, default=10.)
    parser.add_argument('-t', '--tth-max', help="max tth for rings", type=float, default=np.nan)
    parser.add_argument('-o', '--ome-tol', help="omega tolerance", type=float, default=0.5)
    parser.add_argument('-m', '--make-imageseries', help='make imageseries on the fly', action='store_true', default=False)

    args = parser.parse_args()

    instrument_cfg = args.instrument_cfg
    imageseries_file = args.imageseries_file
    materials_file = args.materials_file
    materials_key = args.materials_key
    grains_file = args.grains_file
    grain_ids_arg = args.grain_ids

    slider_delta = args.slider_delta
    tth_max = args.tth_max
    ome_tol = args.ome_tol
    make_images = args.make_imageseries

    # !!! HARD CODED OMEGA RANGE !!!
    ome_range = np.radians([-90, 90])

    # load instrument and imageseries
    instr = load_instrument(instrument_cfg)

    # load plane data
    if np.isnan(tth_max):
        tth_max = None
    pdata = load_pdata(materials_file, materials_key, tth_max=tth_max)

    # load grains table
    gtable = np.loadtxt(grains_file)

    if grain_ids_arg.lower() in ['none', '']:
        grain_ids = None
    else:
        grain_ids = np.array(grain_ids_arg.split(','), dtype=int)

    if make_images:
        if ',' in imageseries_file:
            fnames = imageseries_file.split(',')
        else:
            fnames = glob.glob(imageseries_file)
        print "found the folowing files"
        ims_in = {}
        for fname in fnames:
            print "\t%s" % fname
            ims = imageseries.open(fname, 'frame-cache')
            m = ims.metadata
            ims_in[str(m['panel'])] = ims
    else:
        print "loading single multipanel imageseries %s" % imageseries_file
        make_images = False
        ims_in = load_images(imageseries_file)

    gplist = np.atleast_2d(gtable[:, 3:15])
    iv = IView(instr, ims_in, pdata, gplist,
               grain_ids=grain_ids, make_images=make_images,
               ome_tol=ome_tol, ome_ranges=[ome_range, ])
