#!/usr/bin/env python3

import time
import numpy as np
import datetime
import h5py

import vxi11

MODEL = "DG1032Z"

def setup_pulse(ip,vlow,vhigh,frequency,width):
    """
    """
    instr =  vxi11.Instrument(ip)
    idn = instr.ask("*IDN?")
    print(idn)
    if not (MODEL in idn):
        raise Exception(f"Instrument at {ip} not a {MODEL}, it's a: {idn}")
    instr.write(":source1:function pulse")
    instr.write(f":source1:pulse:period 1e-4")
    #instr.write(f":source1:frequency:center {frequency:f}")
    #instr.write(f":source1:pulse:width {width:f}")
    #instr.write(f":source1:voltage:low {vlow:f}")
    #instr.write(f":source1:voltage:high {vhigh:f}")

if __name__ == "__main__":
    ip = "192.168.55.3"

    setup_pulse(ip,0,3,1e4,20e-9)
