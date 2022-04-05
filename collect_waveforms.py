#!/usr/bin/env python3

from oscilloscope import *
import time
import numpy as np
import datetime
import h5py

import vxi11

def pulser_waveform_run(ip):
    channel="channel1"
    nWaveforms=10000
    now = datetime.datetime.now().replace(microsecond=0)
    setup_vert(ip,200e-3,-400e-3,probe=1,bwlimit="20M",channel=channel)
    setup_horiz(ip,50e-9,0)
    setup_trig(ip,200e-3,10e-6,sweep="single",channel=channel)

    print(f"Collecting {nWaveforms}")
    out_file_name = "waveforms_{}_{:d}waveforms.hdf5".format(now.isoformat(),nWaveforms)
    print(f"Output filename is: {out_file_name}")
    with h5py.File(out_file_name,"w") as out_file:
        collect_waveforms(ip,out_file,nWaveforms,source=channel)

if __name__ == "__main__":
    ip = "192.168.55.2"
    pulser_waveform_run(ip)
