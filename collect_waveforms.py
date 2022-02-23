#!/usr/bin/env python3

from oscilloscope import *
import time
import numpy as np
import datetime
import h5py

import vxi11

def collect_waveforms(ip,out_file_name,nwaveforms,source="channel1"):
    instr =  vxi11.Instrument(ip)
    idn = instr.ask("*IDN?")
    if not (MODEL in idn):
        raise Exception(f"Instrument at {ip} not a {MODEL}, it's a: {idn}")
    instr.write(f":waveform:source {source}")
    instr.write(":waveform:mode raw")
    instr.write(":waveform:format byte")
    #instr.write(":waveform:points 1000")

    instr.write(":stop")

    xorigin = float(instr.ask(":waveform:xorigin?"))
    xincrement = float(instr.ask(":waveform:xincrement?"))
    yorigin = float(instr.ask(":waveform:yorigin?"))
    yreference = float(instr.ask(":waveform:yreference?"))
    yincrement = float(instr.ask(":waveform:yincrement?"))
    waveform_length = int(instr.ask(":waveform:points?"))
    #print(f":waveform:points? is {waveform_length}")

    time_raw = np.arange(waveform_length)
    time_calib = time_raw*xincrement+xorigin

    with h5py.File(out_file_name,"w") as out_file:
        time_ds = out_file.create_dataset("time",data=time_calib)
        time_ds.attrs["units"] = "s"
        time_ds.make_scale("sample time")
        waveforms = out_file.create_dataset("waveforms",(nwaveforms,waveform_length))
        waveforms.attrs["units"] = "V"
        waveforms.dims[0].label = "waveform number"
        waveforms.dims[1].label = "time"
        waveforms.dims[1].attach_scale(time_ds)
        for i in range(nwaveforms):
            instr.write(":single")
            time.sleep(0.1)
            instr.ask(":trigger:status?")
            time.sleep(0.1)
            trigger_status = instr.ask(":trigger:status?")
            if trigger_status != "STOP":
                raise Exception(f"Trigger status should be STOP for raw waveform reading, not {trigger_status}")
            data_raw = instr.ask_raw(":waveform:data?".encode("ASCII"))
            if data_raw[:1] != b"#":
                raise Exception("Data should start with #")
            nbytes_size = int(data_raw[1:2].decode("ascii"))
            size = int(data_raw[2:2+nbytes_size].decode("ascii"))
            data_start = 2+nbytes_size
            data_endp1 = data_start+size
            if size != waveform_length:
                raise Exception(f"Data payload size {size} != {waveform_length} waveform_length")
            data_raw = np.frombuffer(data_raw[data_start:data_endp1],dtype=np.uint8)
            data = (data_raw*1.-yreference-yorigin)*yincrement
            waveforms[i,:] = data

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
    collect_waveforms(ip,out_file_name,nWaveforms,source=channel)

if __name__ == "__main__":
    ip = "192.168.55.2"
    pulser_waveform_run(ip)
