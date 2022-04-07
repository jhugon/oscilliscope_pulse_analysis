#!/usr/bin/env python3

import vxi11
import numpy as np
import h5py
import time
import datetime

MODEL = "MSO5354"

def find_smallest_setting(x):
    """
    Finds the smallest value in the 1-2-5 series that is larger than the given value

    Works with scalars or numpy arrays
    """
    exp = np.floor(np.log10(x))
    significand = x/10**exp
    s_lt2 = significand <= 2.
    s_lt5 = significand <= 5.
    s_gt5 = significand > 5.
    result_significand = 2.*s_lt2 + 5.*np.logical_xor(s_lt5,s_lt2) + 10.*s_gt5
    return result_significand*10**exp


def setup_vert(ip,scale,offset,probe=10,coupling="dc",bwlimit="off",channel="channel1"):
    """
    scale and offset are floats in volts
    probe is the int multiplier for the probe, so 10 for a 10x probe
    coupling is dc or ac
    bwlimit is off, 20M, 100M, or 200M
    """
    instr =  vxi11.Instrument(ip)
    idn = instr.ask("*IDN?")
    if not (MODEL in idn):
        raise Exception(f"Instrument at {ip} not a {MODEL}, it's a: {idn}")
    instr.write(f":{channel}:display on")
    instr.write(f":{channel}:scale {scale:g}") # 200 mV
    instr.write(f":{channel}:offset {offset:g}")
    instr.write(f":{channel}:probe {probe:d}")
    instr.write(f":{channel}:coupling {coupling}")
    instr.write(f":{channel}:bwlimit {bwlimit}") # off 20M 100M 200M


def setup_horiz(ip,scale,offset):
    """
    scale and offset are floats in seconds
    """
    instr =  vxi11.Instrument(ip)
    idn = instr.ask("*IDN?")
    if not (MODEL in idn):
        raise Exception(f"Instrument at {ip} not a {MODEL}, it's a: {idn}")
    instr.write(":timebase:delay:enable off")
    instr.write(f":timebase:scale {scale:g}")
    instr.write(f":timebase:offset {offset:g}")
    instr.write(":timebase:mode main")
    instr.write(":timebase:href:mode center")
    instr.write(":timebase:href:position 0")

def setup_trig(ip,level,holdoff,sweep="normal",channel="channel1"):
    """
    assumes edge mode for now

    level: float trigger level in volts
    holdoff: amount of time to holdoff from triggering
    sweep: trigger mode: auto, normal, or single
    """
    instr =  vxi11.Instrument(ip)
    idn = instr.ask("*IDN?")
    if not (MODEL in idn):
        raise Exception(f"Instrument at {ip} not a {MODEL}, it's a: {idn}")
    instr.write(f":trigger:sweep {sweep}")
    instr.write(":trigger:coupling dc")
    instr.write(f":trigger:holdoff {holdoff}")
    instr.write(":trigger:nreject off") # noise rejection
    instr.write(":trigger:mode edge")
    instr.write(f":trigger:edge:source {channel}")
    instr.write(":trigger:edge:slope positive")
    instr.write(f":trigger:edge:level {level:g}")

def collect_waveforms(ip,out_file,nwaveforms,channel="channel1"):
    """
    Collects a bunch of triggers worth of waveforms, saving to the open h5py
    file or directory given in out_file using oscilliscope at ip, channel. 
    The oscilliscope channel is assumed to be already setup how you want.
    """
    instr =  vxi11.Instrument(ip)
    idn = instr.ask("*IDN?")
    if not (MODEL in idn):
        raise Exception(f"Instrument at {ip} not a {MODEL}, it's a: {idn}")
    instr.write(f":waveform:source {channel}")
    instr.write(":waveform:mode raw")
    instr.write(":waveform:format byte")

    instr.write(":stop")

    xorigin = float(instr.ask(":waveform:xorigin?"))
    xincrement = float(instr.ask(":waveform:xincrement?"))
    yorigin = float(instr.ask(":waveform:yorigin?"))
    yreference = float(instr.ask(":waveform:yreference?"))
    yincrement = float(instr.ask(":waveform:yincrement?"))
    waveform_length = int(instr.ask(":waveform:points?"))

    time_raw = np.arange(waveform_length)
    time_calib = time_raw*xincrement+xorigin

    time_ds = out_file.create_dataset("time",data=time_calib)
    time_ds.attrs["units"] = "s"
    time_ds.attrs["sample_spacing"] = xincrement
    time_ds.attrs["sample_frequency"] = 1./xincrement
    time_ds.make_scale("sample time")
    waveforms = out_file.create_dataset("waveforms_raw",(nwaveforms,waveform_length),dtype=np.uint8)
    waveforms.attrs["help"] = "Change waveforms to float, multiply by calib_slope and add calib_intercept to get waveform values in V"
    waveforms.attrs["calib_slope"] = yincrement
    waveforms.attrs["calib_intercept"] = -yincrement*(yreference+yorigin)
    waveforms.attrs["calib_min"] = waveforms.attrs["calib_intercept"]
    waveforms.attrs["calib_max"] = waveforms.attrs["calib_intercept"]+255*yincrement
    waveforms.attrs["calib_N_steps"] = 255
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
        waveforms[i,:] = data_raw


def setup_sig_gen(ip,source,function,Vpp,offset,frequency,phase=0.,symmetry=50.,duty_cycle=50.,out50Ohm=False):
    instr =  vxi11.Instrument(ip)
    idn = instr.ask("*IDN?")
    if not (MODEL in idn):
        raise Exception(f"Instrument at {ip} not a {MODEL}, it's a: {idn}")
    instr.write(f":source{source}:output:state off")
    instr.write(f":source{source}:function {function}")
    if function.lower() == "ramp":
        instr.write(f":source{source}:function:ramp:symmetry {symmetry:g}")
    if function.lower()[:3] == "squ" or function.lower()[:4] == "puls":
        instr.write(f":source{source}:pulse:dcycle {duty_cycle:g}")
    instr.write(f":source{source}:frequency {frequency:g}")
    instr.write(f":source{source}:phase {phase:g}")
    instr.write(f":source{source}:voltage {Vpp:g}")
    instr.write(f":source{source}:voltage:offset {offset:g}")
    instr.write(f":source{source}:type none") # none, mod, sweep, burst
    if out50Ohm:
        instr.write(f":source{source}:output:impedance fifty") # 50 Ohm
    else:
        instr.write(f":source{source}:output:impedance omeg") # High Z
    instr.write(f":source{source}:output:state on")

if __name__ == "__main__":

    ip = "192.168.55.2"
    channel="channel1"
    nWaveforms=20
    now = datetime.datetime.now().replace(microsecond=0)
    setup_vert(ip,100e-3,0,probe=1,channel=channel)
    setup_horiz(ip,1e-6,0)
    setup_trig(ip,0,10e-6,sweep="single",channel=channel)

    print(f"Collecting {nWaveforms}")
    out_file_name = "dummy_waveforms_{}_{:d}waveforms.hdf5".format(now.isoformat(),nWaveforms)
    print(f"Output filename is: {out_file_name}")
    with h5py.File(out_file_name,"w") as out_file:
        collect_waveforms(ip,out_file,nWaveforms,channel=channel)
