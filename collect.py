#!/usr/bin/env python3

import time
import numpy as np
import datetime
import h5py

import vxi11

MODEL = "MSO5354"

def setup_vert(ip):
    instr =  vxi11.Instrument(ip)
    idn = instr.ask("*IDN?")
    if not (MODEL in idn):
        raise Exception(f"Instrument at ")
    instr.write(":channel1:display on")
    instr.write(":channel1:scale 200e-3") # 200 mV
    instr.write(":channel1:offset 0")
    instr.write(":channel1:probe 1")
    instr.write(":channel1:coupling dc")
    instr.write(":channel1:bwlimit 20M") # off 20M 100M 200M


def setup_horiz(ip):
    instr =  vxi11.Instrument(ip)
    idn = instr.ask("*IDN?")
    if not (MODEL in idn):
        raise Exception(f"Instrument at ")
    instr.write(":timebase:delay:enable off")
    instr.write(":timebase:scale 100e-9") # 100 ns
    instr.write(":timebase:offset 0")
    instr.write(":timebase:mode main")
    instr.write(":timebase:href:mode center")
    instr.write(":timebase:href:position 0")

def setup_trig(ip):
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
    instr.write(":trigger:edge:level 12e-3")

def collect_counter_data(ip):
    instr =  vxi11.Instrument(ip)
    idn = instr.ask("*IDN?")
    if not (MODEL in idn):
        raise Exception(f"Instrument at ")
    instr.write(":counter:enable on")
    instr.write(":counter:source channel1")
    instr.write(":counter:mode totalize") # frequency period totalize
    #trigger_values = np.linspace(-2,15,17*2)
    trig_max = 600
    trig_min = 300
    #trig_n_vals = (trig_max-trig_min)*4+1
    trig_n_vals = 11
    trigger_values = np.linspace(trig_min,trig_max,trig_n_vals)
    time_per_trig_val = 5
    print(f"Spending {time_per_trig_val} s triggering on each of {trig_n_vals} values between {trig_min} and {trig_max} mV")
    now = datetime.datetime.now().replace(microsecond=0)
    out_file_name = "raw_{}.hdf5".format(now.isoformat())
    print(f"Output filename is: {out_file_name}")
    with h5py.File(out_file_name,"w") as out_file:
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

if __name__ == "__main__":
    ip = "192.168.55.2"
    setup_vert(ip)
    setup_horiz(ip)
    setup_trig(ip)
    collect_counter_data(ip)
