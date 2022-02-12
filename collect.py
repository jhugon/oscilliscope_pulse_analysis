#!/usr/bin/env python3

import time
import numpy as np
import datetime
import h5py

ip = "192.168.55.2"

import vxi11
instr =  vxi11.Instrument(ip)
idn = instr.ask("*IDN?")
## write, read, and ask: write then read

if "MSO5354" in idn:
    print("Preparing...")
    instr.write(":channel1:display on")
    instr.write(":channel1:scale 5e-3") # 5 mV
    instr.write(":channel1:offset 0")
    instr.write(":channel1:probe 1")
    instr.write(":channel1:coupling dc")
    instr.write(":channel1:bwlimit off")

    instr.write(":timebase:delay:enable off")
    instr.write(":timebase:scale 100e-9") # 100 ns
    instr.write(":timebase:offset 0")
    instr.write(":timebase:mode main")
    instr.write(":timebase:href:mode center")
    instr.write(":timebase:href:position 0")

    
    instr.write(":trigger:sweep normal") # auto normal single
    instr.write(":trigger:coupling dc")
    instr.write(":trigger:holdoff 8e-9")
    instr.write(":trigger:nreject off") # noise rejection
    instr.write(":trigger:mode edge")
    instr.write(":trigger:edge:source channel1")
    instr.write(":trigger:edge:slope positive")
    instr.write(":trigger:edge:level 12e-3")

    instr.write(":counter:enable on")
    instr.write(":counter:source channel1")
    instr.write(":counter:mode totalize") # frequency period totalize
    #trigger_values = np.linspace(-2,15,17*2)
    trig_max = 8
    trig_min = 0
    trig_n_vals = (trig_max-trig_min)*2+1
    trigger_values = np.linspace(trig_min,trig_max,trig_n_vals)
    time_per_trig_val = 1
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
else:
    print("Instrument isn't MSO5354")
