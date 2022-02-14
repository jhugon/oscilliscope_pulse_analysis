#!/usr/bin/env python3

import time
import numpy as np
import datetime
import h5py

import vxi11

MODEL = "MSO5354"

def setup_vert(ip,scale,offset,probe=10,coupling="dc",bwlimit="off"):
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
    instr.write(":channel1:display on")
    instr.write(f":channel1:scale {scale:g}") # 200 mV
    instr.write(f":channel1:offset {offset:g}")
    instr.write(f":channel1:probe {probe:d}")
    instr.write(f":channel1:coupling {coupling}")
    instr.write(f":channel1:bwlimit {bwlimit}") # off 20M 100M 200M


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

def setup_trig(ip,level):
    """
    assumes edge mode for now

    level: float trigger level in volts
    """
    instr =  vxi11.Instrument(ip)
    idn = instr.ask("*IDN?")
    if not (MODEL in idn):
        raise Exception(f"Instrument at ")
    instr.write(":trigger:sweep normal") # auto normal single
    instr.write(":trigger:coupling dc")
    instr.write(":trigger:holdoff 8e-9")
    instr.write(":trigger:nreject off") # noise rejection
    instr.write(":trigger:mode edge")
    instr.write(":trigger:edge:source channel1")
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

    trigger_values_ds = out_file.create_dataset("trigger_values",data=trigger_values)
    trigger_values_ds.attrs["units"] = "mV"
    counts = out_file.create_dataset("counts",len(trigger_values))
    counts.attrs["time_interval"] = time_per_trig_val
    counts.attrs["time_interval_units"] = "s"
    for i, trig_val in enumerate(trigger_values):
        instr.write(":trigger:edge:level {:f}".format(trig_val*1e-3))
        time.sleep(0.5)
        instr.write(":counter:totalize:clear")
        time.sleep(time_per_trig_val)
        count = instr.ask(":counter:current?")
        counts[i] = count
        print("Count for {} mV trigger: {}".format(trig_val,count))

def normal_counts(ip):
    now = datetime.datetime.now().replace(microsecond=0)
    setup_vert(ip,200e-3,-400e-3,probe=1,bwlimit="20M")
    setup_horiz(ip,100e-9,0)
    setup_trig(ip,100e-3)
    time_per_trig_val = 1
    trig_max = 500
    trig_min = 0
    trig_n_vals = 11
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
    setup_trig(ip,100e-3)
    time_per_trig_val = 5
    trigger_values = np.arange(0,500,8)
    trig_max = trigger_values[-1]
    trig_min = trigger_values[0]
    trig_n_vals = len(trigger_values)
    print(f"Spending {time_per_trig_val} s triggering on each of {trig_n_vals} values between {trig_min} and {trig_max} mV")
    out_file_name = "counts_max_res_{}_{:d}trigs_{:.0f}to{:.0f}mV_{:.0f}s.hdf5".format(now.isoformat(),trig_n_vals,trig_min,trig_max,time_per_trig_val)
    print(f"Output filename is: {out_file_name}")
    with h5py.File(out_file_name,"w") as out_file:
        collect_counter_data(ip,out_file,trigger_values,time_per_trig_val)

if __name__ == "__main__":
    ip = "192.168.55.2"
    #normal_counts(ip)
    max_resolution_counts(ip)
