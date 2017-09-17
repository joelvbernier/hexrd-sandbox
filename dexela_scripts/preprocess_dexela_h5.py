import argparse
import glob
import os

import numpy as np

from hexrd.imageseries import omega
import pp_dexela

# =============================================================================
# USER INPUT
# =============================================================================

# panel keys to MATCH INSTRUMENT FILE
panel_keys = ['FF1', 'FF2']
panel_opts = dict.fromkeys(panel_keys)

# !!!: hard coded options for each dexela for April 2017
panel_opts['FF1'] = [('flip', 'v'), ]
panel_opts['FF2'] = [('flip', 'h'), ]

# ==================== End Inputs (should not need to alter below this line)


def process_dexelas(file_names, samp_name, scan_number,
                    ostart, ostep, num_frames,
                    panel_opts, threshold):
    """
    wrapper for F2 dexela setup
    """
    ostop = ostart + num_frames*ostep
    omw = omega.OmegaWedges(num_frames)
    omw.addwedge(ostart, ostop, num_frames)

    for file_name in file_names:
        for key in panel_keys:
            if key.lower() in file_name:
                ppd = pp_dexela.PP_Dexela(
                    file_name,
                    omw,
                    panel_opts[key],
                    panel_id=key,
                    frame_start=fstart)
                output_name = samp_name + '_' + \
                    str(scan_number) + '_' + \
                    file_name.split('/')[-1].split('.')[0]
                ppd.save_processed(output_name, threshold)


if __name__ == '__main__':
    #
    #  Run preprocessor
    #
    parser = argparse.ArgumentParser(
        description="pre-process double Dexela images from F2")

    parser.add_argument('base_dir',
                        help="raw data path on chess daq", type=str)
    parser.add_argument('expt_name',
                        help="experiment name", type=str)
    parser.add_argument('samp_name',
                        help="sample name", type=str)
    parser.add_argument('scan_number',
                        help="ff scan number", type=int)

    parser.add_argument('-n', '--num-frames',
                        help="number of frames to read",
                        type=int, default=1441)
    parser.add_argument('-s', '--start-frame',
                        help="index of first data frame",
                        type=int, default=4)
    parser.add_argument('-t', '--threshold',
                        help="threshold for frame caches",
                        type=int, default=50)
    parser.add_argument('-o', '--ome-start',
                        help="start omega",
                        type=float, default=0.)
    parser.add_argument('-d', '--ome-delta',
                        help="delta omega",
                        type=float, default=0.2498265093684941)

    args = parser.parse_args()

    # strip args
    data_dir = args.base_dir
    expt_name = args.expt_name
    samp_name = args.samp_name
    scan_number = args.scan_number
    num_frames = args.num_frames
    fstart = args.start_frame
    threshold = args.threshold
    ostart = args.ome_start
    ostep = args.ome_delta

    file_names = glob.glob(
        os.path.join(
            data_dir,
            expt_name,
            samp_name,
            str(scan_number),
            'ff',
            '*.h5')
    )
    check_files_exist = [os.path.exists(file_name) for file_name in file_names]
    if not np.all(check_files_exist):
        raise RuntimeError("files don't exist!")

    process_dexelas(file_names, samp_name, scan_number,
                    ostart, ostep, num_frames,
                    panel_opts, threshold)
