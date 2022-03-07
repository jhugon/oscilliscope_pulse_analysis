#!/usr/bin/env python3

import sys
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
from scipy import fft
from scipy import signal
#from scipy import signal.windows
import matplotlib.pyplot as mpl
import h5py
from hist import Hist
import boost_histogram as bh
import matplotlib.colors

in_file_name = "waveforms_2022-02-19T16:32:50_10000waveforms.hdf5"
with h5py.File(in_file_name) as in_file:
    waveforms_dset = in_file["waveforms"]
    waveform_units = waveforms_dset.attrs["units"]
    ts = waveforms_dset.dims[1][0]
    waveforms = waveforms_dset[:,:][np.amax(waveforms_dset,axis=1)!=0,:]
    nWaveforms, waveform_len = waveforms.shape
    ts_broadcast, _ = np.broadcast_arrays(ts,waveforms)
    ts_units = ts.attrs["units"]
    sample_period = ts[1]-ts[0]
    sample_frequency = 1./sample_period
    print(f"{waveforms_dset.shape[0]} waveforms ({nWaveforms} selected), each of {waveform_len} samples at sample frequency: {sample_frequency:.6g} Hz")


    ## delay line pulse shaping
    delay_line_n = 10*8 # *8 for ns
    waveforms_delay = np.zeros(waveforms.shape)
    for iWaveform in range(nWaveforms):
        waveforms_delay[iWaveform,:] = np.roll(waveforms[iWaveform,:],delay_line_n)
    waveforms_delay_shaped = waveforms[:,:]-waveforms_delay

    ## Trapezoidal pulse shaping--signal pulses seem too short for this (at least compared to rise time)
    trap_n_avg = 10*8
    trap_n_gap = 10*8
    waveforms_cumsum = np.cumsum(waveforms[:,:],axis=1)
    ## this is a forward looking moving average
    waveforms_sma = np.array(waveforms)
    waveforms_sma[:,:-trap_n_avg] = -waveforms_cumsum[:,:-trap_n_avg]+waveforms_cumsum[:,trap_n_avg:]
    waveforms_sma[:,:-trap_n_avg] = waveforms_sma[:,:-trap_n_avg] / trap_n_avg
    waveforms_trap = np.array(waveforms_sma)
    waveforms_trap[:,:-(trap_n_avg+trap_n_gap)] = waveforms_sma[:,trap_n_avg+trap_n_gap:]-waveforms_sma[:,:-(trap_n_avg+trap_n_gap)]

    fig, ax = mpl.subplots(figsize=(6,6),constrained_layout=True)
    for i in range(min(nWaveforms,1)):
        ax.plot(ts[:]*1e9,waveforms[i,:]*1e3,label="Un-processed")
        #ax.plot(ts[:]*1e9,waveforms_delay[i,:]*1e3,label="Delay")
        ax.plot(ts[:]*1e9,waveforms_delay_shaped[i,:]*1e3,label="Delay-line shaped")
        ax.plot(ts[:]*1e9,waveforms_cumsum[i,:],label=f"cumsum")
        ax.plot(ts[:]*1e9,waveforms_sma[i,:]*1e3,label=f"SMA {trap_n_avg} samples")
        ax.plot(ts[:]*1e9,waveforms_trap[i,:]*1e3,label=f"Trap-shaped {trap_n_avg}/{trap_n_gap} samples r-f/g")
    ax.set_xlabel(f"Time [n{ts_units}]")
    ax.set_ylabel(f"Waveform [m{waveform_units}]")
    ax.legend()
    fig.savefig("alt_waveform.png")
    fig.savefig("alt_waveform.pdf")
