from __future__ import print_function

import argparse
import os
import glob

import numpy as np

from hexrd.imageseries import omega

import pp_dexela

# RAW FILES DIRECTORY TEMPLATE FOR CHESS DAQ
CHESS_BASE = '/nfs/chess/raw/current/f2/pokharel-603-1/%s/%d/ff/*.h5'

def save_processed(file_names, panel_keys, omw, fstart, threshold):
    for file_name in file_names:
        for key in panel_keys:
            if key.lower() in file_name:
                ppd = pp_dexela.PP_Dexela(
                    file_name, 
                    omw, 
                    panel_opts[key], 
                    frame_start=fstart)
                output_name = file_name.split('/')[-1].split('.')[0]
                ppd.save_processed(output_name, threshold)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="preprocess dexelas from raw h5 to frame-caches")
    
    parser.add_argument('sample_name', help="sample/raw directory name", type=str)
    parser.add_argument('scan_number', help="scan number", type=int)
    
    parser.add_argument('-b', '--base-dir-str', help="base directory template", type=int, default=CHESS_BASE)
    parser.add_argument('-f', '--first-frame', help="first frame index, 0-based", type=int, default=4)
    parser.add_argument('-t', '--threshold', help="threshold value for frame cache", type=int, default=400)
    parser.add_argument('-s', '--omega-start', help="starting omega in degrees", type=float, default=0.)
    parser.add_argument('-d', '--omega-step', help="omega step in degrees", type=float, default=0.25)
    parser.add_argument('-n', '--num-frames', help="number of frames", type=int, default=1440)
    
    args = parser.parse_args()
    
    sample_name = args.sample_name
    scan_number = args.scan_number
    
    fstart = args.first_frame
    threshold = args.threshold
    ostart = args.omega_start
    ostep = args.omega_step
    nframes = args.num_frames
        
    file_names = glob.glob(CHESS_BASE % (sample_name, scan_number))

    check_files_exist = [os.path.exists(file_name) for file_name in file_names]
    if not np.all(check_files_exist):
        raise RuntimeError("files don't exist!")

    # panel keys to MATCH INSTRUMENT FILE
    panel_keys = ['FF1', 'FF2']
    panel_opts = dict.fromkeys(panel_keys)
    
    # !!!: hard coded options for each dexela for April 2017
    panel_opts['FF1'] = [('flip', 'v'), ]
    panel_opts['FF2'] = [('flip', 'h'), ]

    # omega information in DEGREES
    ostop = ostart + nframes*ostep
    omw = omega.OmegaWedges(nframes)
    omw.addwedge(ostart, ostop, nframes)

    print("processing files:")
    for file_name in file_names:
        print("%s" %file_name)
    save_processed(file_names, panel_keys, omw, fstart, threshold)
    

    