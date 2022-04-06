#!/usr/bin/env python3

from oscilloscope import *
import numpy as np
import time
import datetime
import h5py
from hist import Hist
import matplotlib.pyplot as mpl
import matplotlib.colors

import vxi11

def collect_step_response_data(ip,verts_in,verts_sig_gen,nWaveforms,in_channel="channel1",sig_gen_channel="1",horiz=(500e-9,0),trigger_thresholds=None):
    """
    Generates a square waveform and records the resulting input waveforms.
    """
    now = datetime.datetime.now().replace(microsecond=0)
    setup_horiz(ip,*horiz)

    print(f"Collecting {nWaveforms}")
    out_file_name = "step_response_{}_{:d}waveforms.hdf5".format(now.isoformat(),nWaveforms)
    print(f"Output filename is: {out_file_name}")

    if trigger_thresholds is None:
        trigger_thresholds = np.zeros(len(verts_in))

    with h5py.File(out_file_name,"w") as out_file:
        step_response_grp = out_file.create_group("step_response")
        for iVert,(vert_in,vert_sig_gen,trigger_threshold) in enumerate(zip(verts_in,verts_sig_gen,trigger_thresholds)):
            amp_group = step_response_grp.create_group(f"sig_gen_setting{iVert}")
            amp_group.attrs["amplitude"] = vert_in[0]
            amp_group.attrs["offset"] = vert_in[1]
            amp_group.attrs["amplitude_units"] = "V"
            amp_group.attrs["offset_units"] = "V"
            setup_vert(ip,vert_in[0],vert_in[1],probe=1,channel=in_channel)
            setup_trig(ip,trigger_threshold,10e-6,sweep="single",channel=in_channel)
            setup_sig_gen(ip,sig_gen_channel,"square",vert_sig_gen[0],vert_sig_gen[1],2e5,out50Ohm=True)
            time.sleep(0.5)
            collect_waveforms(ip,amp_group,nWaveforms,channel=in_channel)

    return out_file_name


def collect_positive_step_response_data(ip,amplitudes_sig_gen,nWaveforms,in_channel="channel1",sig_gen_channel="1",horiz=(500e-9,0)):
    verts_sig_gen = [(x,0.5*x) for x in amplitudes_sig_gen]
    scale_in = find_smallest_setting(np.array(amplitudes_sig_gen)*1.1/4.)
    verts_in = [(x,-0.5*y) for x,y in zip(scale_in,amplitudes_sig_gen)]
    trigger_thresholds = [0.5*x for x in amplitudes_sig_gen]
    return collect_step_response_data(ip,verts_in,verts_sig_gen,nWaveforms,in_channel=in_channel,sig_gen_channel=sig_gen_channel,horiz=horiz,trigger_thresholds=trigger_thresholds)


def collect_bipolar_step_response_data(ip,amplitudes_sig_gen,nWaveforms,in_channel="channel1",sig_gen_channel="1",horiz=(500e-9,0)):
    verts_sig_gen = [(2*x,0.) for x in amplitudes_sig_gen]
    scale_in = find_smallest_setting(np.array(amplitudes_sig_gen)*1.1/4.)
    verts_in = [(x,0.) for x in scale_in]
    return collect_step_response_data(ip,verts_in,verts_sig_gen,nWaveforms,in_channel=in_channel,sig_gen_channel=sig_gen_channel,horiz=horiz)


def analyze_step_response_data(fn):

    with h5py.File(fn) as f:
        sr_dir = f["step_response"]
        sig_gen_Vpp_values = []
        for sgs_key in sr_dir:
            sgs_dir = sr_dir[sgs_key]
            sig_gen_Vpp = sgs_dir.attrs["amplitude"]
            sig_gen_Vpp_values.append(sig_gen_Vpp)
        print(sig_gen_Vpp_values)
        waveform_hist = Hist.new.Variable(sig_gen_Vpp_values+[sig_gen_Vpp_values[-1]*2],name="sig_gen_Vpp",label="Signal Generator $V_{pp}$ [V]").Reg(500,-0.6,0.6,name="waveform",label="Waveform [V]").Double()
        for sgs_key in sr_dir:
            sgs_dir = sr_dir[sgs_key]
            sig_gen_Vpp = sgs_dir.attrs["amplitude"]
            waveforms_dset = sgs_dir["waveforms"]
            ts = waveforms_dset.dims[1][0]
            waveform_hist.fill(sig_gen_Vpp,waveforms_dset[:,:].flatten())
        fig, ax = mpl.subplots(figsize=(6,6),constrained_layout=False)
        waveform_hist.plot2d(ax=ax,norm=matplotlib.colors.PowerNorm(gamma=0.3))#,vmax=5000))
        ax.set_title("Recorded Waveform Values")
        fig.savefig("step_response_waveform_hist.png")
        fig.savefig("step_response_waveform_hist.pdf")
        

if __name__ == "__main__":
    ip = "192.168.55.2"
    #out_fn = collect_positive_step_response_data(ip,[0.01,0.03,0.05,0.1,0.3,0.5],20)
    fn = "step_response_2022-04-06T16:07:16_20waveforms.hdf5"
    analyze_step_response_data(fn)
