#!/usr/bin/env python3

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
