#!/usr/bin/env python3

from oscilloscope import *
import time
import numpy as np
import datetime
import h5py

import vxi11

def collect_step_response_data(ip,nWaveforms):
    """
    Generates a square waveform and records the resulting input waveforms.
    """
    channel="channel1"
    sig_gen_channel="1"
    now = datetime.datetime.now().replace(microsecond=0)
    setup_vert(ip,5,0,probe=1,channel=channel)
    setup_horiz(ip,1e-6,0)
    setup_trig(ip,0,10e-6,sweep="single",channel=channel)

    print(f"Collecting {nWaveforms}")
    out_file_name = "step_response_{}_{:d}waveforms.hdf5".format(now.isoformat(),nWaveforms)
    print(f"Output filename is: {out_file_name}")

    amplitudes = [0.01,0.1] # in volts

    with h5py.File(out_file_name,"w") as out_file:
        step_response_grp = out_file.create_group("step_response")
        for iAmplitude,amplitude in enumerate(amplitudes):
            amp_group = step_response_grp.create_group(f"amplitude{iAmplitude}")
            amp_group.attrs["amplitude"] = amplitude
            amp_group.attrs["amplitude_units"] = "V"
            setup_sig_gen(ip,sig_gen_channel,"square",amplitude,0,1e-7,out50Ohm=True)
            collect_waveforms(ip,amp_group,nWaveforms,source=channel)

    return out_file_name

if __name__ == "__main__":
    ip = "192.168.55.2"
    out_fn = collect_step_response_data(ip,10)
