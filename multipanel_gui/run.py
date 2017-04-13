import sys

import cPickle
import yaml
import numpy as np

from hexrd import imageseries
from hexrd import instrument
from calibrate import InstrumentViewer as IView

# plane data
def load_pdata(cpkl):
    with file(cpkl, "r") as matf:
        matlist = cPickle.load(matf)
    return matlist[0].planeData

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
    instr = load_instrument(sys.argv[1])
    ims = load_images(sys.argv[2])
    pdata = load_pdata(sys.argv[3])

    iv = IView(instr, ims, pdata)
