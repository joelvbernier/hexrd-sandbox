import sys
import argparse
import cPickle
import yaml
import numpy as np

from hexrd import imageseries
from hexrd import instrument
from calibrate import InstrumentViewer as IView

# plane data
def load_pdata(cpkl, key, tth_max=None):
    """
    tth_max is in DEGREES
    """
    with file(cpkl, "r") as matf:
        matlist = cPickle.load(matf)
    pd = dict(zip([i.name for i in matlist], matlist))[key].planeData
    if tth_max is not None:
        pd.tThMax = np.radians(tth_max)
    return pd

# images
def load_images(filename):
    return imageseries.open(filename, "hdf5", path='/imageseries')

# instrument
def load_instrument(yml):
    with file(yml, 'r') as f:
        icfg = yaml.load(f)
    return instrument.HEDMInstrument(instrument_config=icfg)

if __name__ == '__main__':
    #
    #  Run viewer
    #
    parser = argparse.ArgumentParser(description="plot rings over an interactive, renedered multipanel imageseries")
    
    parser.add_argument('instrument_cfg', help="instrument config YAML file", type=str)
    parser.add_argument('imageseries_file', help="multipanel imageseries file", type=str)
    parser.add_argument('materials_file', help="materials database", type=str)
    parser.add_argument('materials_key', help="key for material in database", type=str)
    
    parser.add_argument('-d', '--slider-delta', help="+/- delta for slider range", type=float, default=10.)
    parser.add_argument('-p', '--plane-distance', help="distance of projection plane downstream", type=float, default=1000)
    parser.add_argument('-t', '--tth-max', help="max tth for rings", type=float, default=np.nan)
    
    args = parser.parse_args()
   
    instrument_cfg = args.instrument_cfg
    imageseries_file = args.imageseries_file
    materials_file = args.materials_file
    materials_key = args.materials_key
    slider_delta = args.slider_delta
    plane_distance = args.plane_distance
    tth_max = args.tth_max
    
    # load instrument and imageseries
    instr = load_instrument(instrument_cfg)
    ims = load_images(imageseries_file)
    
    # load plane data
    if np.isnan(tth_max):
        tth_max = None
    pdata = load_pdata(materials_file, materials_key, tth_max=tth_max)
 
    tvec = np.r_[0., 0., -plane_distance]

    iv = IView(instr, ims, pdata, tvec=tvec, slider_delta=slider_delta)
