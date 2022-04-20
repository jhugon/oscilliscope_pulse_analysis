#!/usr/bin/env python3

import vxi11
import numpy as np
import h5py
import time
import datetime

MODEL = "MSO5354"

def auto_scale(ip):
    instr =  vxi11.Instrument(ip)
    idn = instr.ask("*IDN?")
    if not (MODEL in idn):
        raise Exception(f"Instrument at {ip} not a {MODEL}, it's a: {idn}")
    instr.write(":autoscale")

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


def disable_channel(ip,channel="channel1"):
    instr =  vxi11.Instrument(ip)
    idn = instr.ask("*IDN?")
    if not (MODEL in idn):
        raise Exception(f"Instrument at {ip} not a {MODEL}, it's a: {idn}")
    instr.write(f":{channel}:display off")


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

def get_oscilloscope_settings(instr,channel="channel1"):
    idn = instr.ask("*IDN?")
    if not (MODEL in idn):
        raise Exception(f"Instrument at {ip} not a {MODEL}, it's a: {idn}")

    result = {}

    result["channel display"] = instr.ask(f":{channel}:display?")
    result["vertical scale"] = instr.ask(f":{channel}:scale?")
    result["vertical offset"] = instr.ask(f":{channel}:offset?")
    result["channel probe"] = instr.ask(f":{channel}:probe?")
    result["channel coupling"] = instr.ask(f":{channel}:coupling?")
    result["channel bwlimit"] = instr.ask(f":{channel}:bwlimit?")

    result["time scale"] = instr.ask(":timebase:scale?")
    result["time offset"] = instr.ask(":timebase:offset?")

    result["trigger coupling"] = instr.ask(":trigger:coupling?")
    result["trigger holdoff"] = instr.ask(":trigger:holdoff?")
    result["trigger noise rejection"] = instr.ask(":trigger:nreject?") # noise rejection
    result["trigger mode"] = instr.ask(":trigger:mode?")
    result["trigger edge source"] = instr.ask(":trigger:edge:source?")
    result["trigger edge slope"] = instr.ask(":trigger:edge:slope?")
    result["trigger edge level"] = instr.ask(":trigger:edge:level?")
    result["trigger threshold"] = result["trigger edge level"]

    return result


def collect_waveforms(ip,out_file,nwaveforms,channel="channel1",nRetries=3):
    """
    Collects a bunch of triggers worth of waveforms, saving to the open h5py
    file or directory given in out_file using oscilliscope at ip, channel. 
    The oscilliscope channel is assumed to be already setup how you want.
    """
    instr =  vxi11.Instrument(ip)
    idn = instr.ask("*IDN?")
    if not (MODEL in idn):
        raise Exception(f"Instrument at {ip} not a {MODEL}, it's a: {idn}")

    oscilliscope_settings = get_oscilloscope_settings(instr,channel)

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
    waveforms.attrs.update(oscilliscope_settings)
    waveforms.attrs["help"] = "Change waveforms to float, multiply by calib_slope and add calib_intercept to get waveform values in V"
    waveforms.attrs["calib_slope"] = yincrement
    waveforms.attrs["calib_intercept"] = -yincrement*(yreference+yorigin)
    waveforms.attrs["calib_min"] = waveforms.attrs["calib_intercept"]
    waveforms.attrs["calib_max"] = waveforms.attrs["calib_intercept"]+255*yincrement
    waveforms.attrs["calib_N_steps"] = 255
    waveforms.attrs["calib_units"] = "V"
    waveforms.dims[0].label = "waveform number"
    waveforms.dims[1].label = "time"
    waveforms.dims[1].attach_scale(time_ds)
    for i in range(nwaveforms):
        iTry = 0
        data_raw = None
        while iTry < nRetries:
            try:
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
                if len(data_raw) != waveform_length:
                    raise Exception(f"Extracted waveform len {len(data_raw)} != {waveform_length} expected waveform_length")
            except Exception as e:
                print(f"Warning on capture of waveform number {i}: {e}",flush=True)
                iTry += 1
            else:
                break
        if data_raw is None:
            raise Exception(f"Couldn't collect waveform number {i} after {nRetries} tries")
        waveforms[i,:] = data_raw


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


def do_measurement(ip,measurement,source="channel1",source2=None,n_avg=2):
    """
    measurement: "vamp", "vpp", "vrms", "frequency", "period", "rtime", "ftime"
    n_avg: number of points to average over
    """
    instr =  vxi11.Instrument(ip)
    idn = instr.ask("*IDN?")
    if not (MODEL in idn):
        raise Exception(f"Instrument at {ip} not a {MODEL}, it's a: {idn}")

    instr.write(f":run")
    instr.write(f":measure:threshold:default")
    instr.write(f":measure:mode precision")
    instr.write(f":measure:clear")
    instr.write(f":measure:statistic:display on")

    if source2 is None:
        instr.write(f":measure:statistic:item {measurement},{source}")
    else:
        instr.write(f":measure:statistic:item {measurement},{source},{source2}")
    instr.write(f":measure:statistic:reset all")
    n_collected = 0
    n_reads = 0
    printed_warning = False
    while n_collected < n_avg:
        time.sleep(1)
        if source2 is None:
            cnt = instr.ask(f":measure:statistic:item? cnt,{measurement},{source}")
        else:
            cnt = instr.ask(f":measure:statistic:item? cnt,{measurement},{source},{source2}")
        n_collected = int(float(cnt))
        n_reads += 1
        if n_reads > 30 and not printed_warning:
            print(f"Warning: measurment taking a while, collected {n_collected} measurement points of {n_avg} required")
            printed_warning = True
    instr.write(f":stop")
    if source2 is None:
        result_val = float(instr.ask(f":measure:statistic:item? averages,{measurement},{source}"))
        result_cnt = int(float(instr.ask(f":measure:statistic:item? cnt,{measurement},{source}")))
        result_std = float(instr.ask(f":measure:statistic:item? deviation,{measurement},{source}"))
        return result_val, result_std, result_cnt
    else:
        result_val = float(instr.ask(f":measure:statistic:item? averages,{measurement},{source},{source2}"))
        result_cnt = int(float(instr.ask(f":measure:statistic:item? cnt,{measurement},{source},{source2}")))
        result_std = float(instr.ask(f":measure:statistic:item? deviation,{measurement},{source},{source2}"))
        return result_val, result_std, result_cnt

def do_measurements(ip,measurements_and_sources,measure_time=2):
    """
    DOESN'T WORK!

    measurements_and_sources should be a list of tuples of measurements and sources like:

    [("vamp","source1"),("vavg","source1")]


    sources: "channel1", "channel2", ...
    measurement: "vamp", "vpp", "vrms", "frequency", "period", "rtime", "ftime"
    """
    assert(len(measurements_and_sources) <= 10)
    instr =  vxi11.Instrument(ip)
    idn = instr.ask("*IDN?")
    if not (MODEL in idn):
        raise Exception(f"Instrument at {ip} not a {MODEL}, it's a: {idn}")

    instr.write(f":run")
    instr.write(f":measure:threshold:default")
    instr.write(f":measure:mode precision")
    instr.write(f":measure:clear")
    instr.write(f":measure:statistic:display on")

    for measurement, sources in measurements_and_sources:
        time.sleep(0.1)
        print(measurement, sources)
        if isinstance(sources,str):
            instr.write(f":measure:statistic:item {measurement},{sources}")
        elif isinstance(sources[0],str) and isinstance(sources[1],str):
            instr.write(f":measure:statistic:item {measurement},{sources[0]},{sources[1]}")
        else:
            raise ValueError(f"Don't know what to do with sources: {sources}")
    instr.write(f":measure:statistic:reset all")
    time.sleep(measure_time)
    instr.write(f":stop")
    results = []
    for measurement, sources in measurements_and_sources:
        if isinstance(sources,str):
            result = [float('nan')]*3
            result[0] = float(instr.ask(f":measure:statistic:item? averages,{measurement},{sources}"))
            result[1] = float(instr.ask(f":measure:statistic:item? deviation,{measurement},{sources}"))
            result[2] = int(float(instr.ask(f":measure:statistic:item? cnt,{measurement},{sources}")))
            results.append(result)
        elif isinstance(sources[0],str) and isinstance(sources[1],str):
            instr.write(f":measure:statistic:item {measurement},{sources[0]},{sources[1]}")
            result = [float('nan')]*3
            result[0] = float(instr.ask(f":measure:statistic:item? averages,{measurement},{sources[0]},{sources[1]}"))
            result[1] = float(instr.ask(f":measure:statistic:item? deviation,{measurement},{sources[0]},{sources[1]}"))
            result[2] = int(float(instr.ask(f":measure:statistic:item? cnt,{measurement},{sources[0]},{sources[1]}")))
            results.append(result)
        else:
            raise ValueError(f"Don't know what to do with sources: {sources}")
    return results


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

def turn_off_sig_gens(ip,sources=["1","2"]):
    instr =  vxi11.Instrument(ip)
    idn = instr.ask("*IDN?")
    if not (MODEL in idn):
        raise Exception(f"Instrument at {ip} not a {MODEL}, it's a: {idn}")
    for source in sources:
        instr.write(f":source{source}:output:state off")

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

    result = do_measurement(ip,"vamp",source="channel1")
    print(result)
