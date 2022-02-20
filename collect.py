#!/usr/bin/env python3

import time
import numpy as np
import datetime
import h5py

import matplotlib.pyplot as mpl

import vxi11

MODEL = "MSO5354"

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
        raise Exception(f"Instrument at ")
    instr.write(f":trigger:sweep {sweep}")
    instr.write(":trigger:coupling dc")
    instr.write(f":trigger:holdoff {holdoff}")
    instr.write(":trigger:nreject off") # noise rejection
    instr.write(":trigger:mode edge")
    instr.write(f":trigger:edge:source {channel}")
    instr.write(":trigger:edge:slope positive")
    instr.write(f":trigger:edge:level {level:g}")

def collect_counter_data(ip,out_file,trigger_values,time_per_trig_val):
    """
    out_file: open h5py file to write to
    trigger_values: a list of trigger levels (in millivolts) to collect counter data for
    time_per_trig_val: how long to count for each trigger value, in seconds
    """
    instr =  vxi11.Instrument(ip)
    idn = instr.ask("*IDN?")
    if not (MODEL in idn):
        raise Exception(f"Instrument at {ip} not a {MODEL}, it's a: {idn}")
    instr.write(":counter:enable on")
    instr.write(":counter:source channel1")
    instr.write(":counter:mode totalize") # frequency period totalize

    trigger_values_set_ds = out_file.create_dataset("trigger_values_set",data=trigger_values)
    trigger_values_set_ds.attrs["units"] = "mV"
    counts = out_file.create_dataset("counts",len(trigger_values))
    counts.attrs["time_interval"] = time_per_trig_val
    counts.attrs["time_interval_units"] = "s"
    trigger_values_ds = out_file.create_dataset("trigger_values",len(trigger_values))
    trigger_values_ds.attrs["units"] = "mV"
    for i, trig_val in enumerate(trigger_values):
        instr.write(":trigger:edge:level {:f}".format(trig_val*1e-3))
        time.sleep(0.1)
        trigger_val_readback = instr.ask(":trigger:edge:level?")
        trigger_val_readback = float(trigger_val_readback)*1e3
        time.sleep(0.1)
        instr.write(":counter:totalize:clear")
        time.sleep(time_per_trig_val)
        count = instr.ask(":counter:current?")
        counts[i] = count
        trigger_values_ds[i] = trigger_val_readback
        print("Count for set {} mV read {:.1f} mV trigger: {}".format(trig_val,trigger_val_readback,count))

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


def normal_counts(ip):
    now = datetime.datetime.now().replace(microsecond=0)
    setup_vert(ip,200e-3,-400e-3,probe=1,bwlimit="20M")
    setup_horiz(ip,100e-9,0)
    setup_trig(ip,100e-3,10e-6)
    time_per_trig_val = 1
    trig_max = 850
    trig_min = 350
    trig_n_vals = 20
    trigger_values = np.linspace(trig_min,trig_max,trig_n_vals)
    print(f"Spending {time_per_trig_val} s triggering on each of {trig_n_vals} values between {trig_min} and {trig_max} mV")
    out_file_name = "counts_{}_{:d}trigs_{:.0f}to{:.0f}mV_{:.0f}s.hdf5".format(now.isoformat(),trig_n_vals,trig_min,trig_max,time_per_trig_val)
    print(f"Output filename is: {out_file_name}")
    with h5py.File(out_file_name,"w") as out_file:
        collect_counter_data(ip,out_file,trigger_values,time_per_trig_val)

def max_resolution_counts(ip):
    """
    trigger must be a multiple of 8 mV
    """
    now = datetime.datetime.now().replace(microsecond=0)
    setup_vert(ip,200e-3,-400e-3,probe=1,bwlimit="20M")
    setup_horiz(ip,100e-9,0)
    setup_trig(ip,100e-3,10e-6)
    time_per_trig_val = 1
    trigger_values = np.arange(344,850,8)
    trig_max = trigger_values[-1]
    trig_min = trigger_values[0]
    trig_n_vals = len(trigger_values)
    print(f"Spending {time_per_trig_val} s triggering on each of {trig_n_vals} values between {trig_min} and {trig_max} mV")
    out_file_name = "counts_max_res_{}_{:d}trigs_{:.0f}to{:.0f}mV_{:.0f}s.hdf5".format(now.isoformat(),trig_n_vals,trig_min,trig_max,time_per_trig_val)
    print(f"Output filename is: {out_file_name}")
    with h5py.File(out_file_name,"w") as out_file:
        collect_counter_data(ip,out_file,trigger_values,time_per_trig_val)

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
    #normal_counts(ip)
    #max_resolution_counts(ip)
    pulser_waveform_run(ip)
