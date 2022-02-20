#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
from scipy import fft
from scipy import signal
#from scipy import signal.windows
import matplotlib.pyplot as mpl
import h5py

freq_cutoff = 200e6 # 50 Mhz

in_file_name = "waveforms_2022-02-19T16:32:50_10000waveforms.hdf5"
in_file_name = "waveforms_2022-02-19T16:17:44_100wavforms.hdf5"
with h5py.File(in_file_name) as in_file:
    waveforms = in_file["waveforms"]
    nWaveforms, waveform_len = waveforms.shape
    waveform_units = waveforms.attrs["units"]
    ts = waveforms.dims[1][0]
    ts_units = ts.attrs["units"]
    sample_period = ts[1]-ts[0]
    sample_frequency = 1./sample_period
    print(f"{nWaveforms} waveforms, each of {waveform_len} samples at sample frequency: {sample_frequency:.6g} Hz")
    print(f"filter cutoff frequency: {freq_cutoff:.6g} Hz, sigma: {1./freq_cutoff:.6g} s, FWHM: {2.355/freq_cutoff} s")

    waveform_ffts = fft.rfft(waveforms[:,:])
    fft_freqs = fft.rfftfreq(waveforms.shape[-1],d=sample_period)
    waveform_fft_amp_mean = np.sum(abs(waveform_ffts),axis=0)/nWaveforms

    gaussian_width = sample_frequency/freq_cutoff
    window = signal.windows.gaussian(waveform_len,std=gaussian_width)
    window = fft.fftshift(window)
    window /= window.sum()
    window_fft = fft.rfft(window)

    waveform_filtered_ffts = waveform_ffts*window_fft
    waveforms_filtered = fft.irfft(waveform_filtered_ffts,waveform_len)

    fig, ax = mpl.subplots(figsize=(6,6),constrained_layout=True)
    for i in range(nWaveforms):
        #ax.plot(ts[:]*1e9,waveforms[i,:]*1e3,label="Unfiltered")
        ax.plot(ts[:]*1e9,waveforms_filtered[i,:]*1e3,label="filtered")
    ax.set_xlabel(f"Time [n{ts_units}]")
    ax.set_ylabel(f"Waveform [m{waveform_units}]")
    #ax.legend()
    fig.savefig("waveform.png")
    fig.savefig("waveform.pdf")

    fig, ax = mpl.subplots(figsize=(6,6),constrained_layout=True)
    #ax.plot(fft.fftshift(fft_freqs),fft.fftshift(abs(waveform_fft_amp_mean)))
    #ax.plot(fft.fftshift(fft_freqs),fft.fftshift(abs(waveform_fft_amp_mean*window_fft)))
    ax.plot(fft_freqs,abs(waveform_fft_amp_mean))
    ax.plot(fft_freqs,abs(waveform_fft_amp_mean*window_fft))
    ax.set_xlabel(f"Frequency [{ts_units}$^{{-1}}$]")
    ax.set_ylabel(f"Waveform Amplitude [{waveform_units}]")
    ax.set_title(f"Average Spectrum Over {nWaveforms} Waveforms")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-2,1e3)

    fig.savefig("waveform_fft.png")
    fig.savefig("waveform_fft.pdf")
