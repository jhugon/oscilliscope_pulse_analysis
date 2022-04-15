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
import zfit

from oscilloscope import *
from waveform_analysis import *


def collect_pulser_waveform_data(ip,nWaveforms):
    channel="channel1"
    trigger_hreshold = 200e-3
    setup_vert(ip,200e-3,-400e-3,probe=1,bwlimit="20M",channel=channel)
    setup_horiz(ip,50e-9,0)
    setup_trig(ip,trigger_hreshold,10e-6,sweep="single",channel=channel)

    print(f"Collecting {nWaveforms}")
    print(f"Ensure oscilloscope {channel} is hooked up to the the DUT output.")
    description = get_description()

    now = datetime.datetime.now().replace(microsecond=0)
    out_file_name = "pulses_{}_{:d}waveforms.hdf5".format(now.isoformat(),nWaveforms)
    print(f"Output filename is: {out_file_name}")
    with h5py.File(out_file_name,"w") as out_file:
        out_file.attrs["trigger threshold"] = trigger_threshold
        out_file.attrs["oscilloscope input channel"] = in_channel
        out_file.attrs["description"] = description
        out_file.attrs["starttime"] = now.isoformat()
        out_file.attrs["status"] = "fail"
        collect_waveforms(ip,out_file,nWaveforms,source=channel)
        out_file.attrs["status"] = "success"
    return out_file_name


def analyze_pulses(in_file_name):
    freq_cutoff = 200e6 # Hz

    with h5py.File(in_file_name) as in_file:
        run_description = in_file.attrs["description"]
        run_starttime = in_file.attrs["starttime"]
        print(f"Run start time: {run_starttime}")
        print(f"Run description: {run_description}")
        waveforms_dset = in_file["waveforms"]
        waveform_units = waveforms_dset.attrs["units"]
        ts = waveforms_dset.dims[1][0]
        waveforms = calibrate_waveforms(waveforms_dset)
        nWaveforms, waveform_len = waveforms.shape
        ts_broadcast, _ = np.broadcast_arrays(ts,waveforms)
        ts_units = ts.attrs["units"]
        sample_period = ts[1]-ts[0]
        sample_frequency = 1./sample_period
        print(f"{waveforms_dset.shape[0]} waveforms ({nWaveforms} selected), each of {waveform_len} samples at sample frequency: {sample_frequency:.6g} Hz")
        print(f"filter cutoff frequency: {freq_cutoff:.6g} Hz, sigma: {1./freq_cutoff:.6g} s, FWHM: {2.355/freq_cutoff} s")

        amax = np.amax(waveforms,axis=1)
        argmax = np.argmax(waveforms,axis=1)

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
        waveform_filtered_hist = Hist.new.Reg(100,-200,200,name="time",label="Time [ns]").Reg(100,-200,900,name="waveform",label="Waveform [mV]").Double()
        waveform_filtered_hist.fill(ts_broadcast[:,:].flatten()*1e9,waveforms_filtered[:,:].flatten()*1e3)
        amax_filtered = np.amax(waveforms_filtered,axis=1)
        argmax_filtered = np.argmax(waveforms_filtered,axis=1)
        argmax_filtered_ts = np.array([ts[x] for x in argmax_filtered])
        select_peak_location = np.logical_and(argmax_filtered_ts > -10e-9,argmax_filtered_ts < 40e-9)
        print(f"filtered pulse peak goes from {min(amax_filtered)*1e3:.1f} to {max(amax_filtered)*1e3:.1f} mV, quartiles: {np.quantile(amax_filtered,0.25)*1e3:.1f}, {np.quantile(amax_filtered,0.5)*1e3:.1f}, {np.quantile(amax_filtered,0.75)*1e3:.1f} mV")
        print(f"filtered pulse peak location goes from index {ts[int(min(argmax_filtered))]*1e9:.1f} to {ts[int(max(argmax_filtered))]*1e9:.1f} ns, quartiles: {ts[int(np.quantile(argmax_filtered,0.25))]*1e9:.1f}, {ts[int(np.quantile(argmax_filtered,0.5))]*1e9:.1f}, {ts[int(np.quantile(argmax_filtered,0.75))]*1e9:.1f} ns")

        waveforms_filtered_shifted = waveforms_filtered[:,:]
        for iWaveform in range(nWaveforms):
            waveforms_filtered_shifted[iWaveform,:] = np.roll(waveforms_filtered_shifted[iWaveform,:],waveform_len//2-argmax_filtered[iWaveform])
        waveforms_filtered_shifted_normalized = (waveforms_filtered_shifted.T/amax_filtered).T

        waveform_filtered_shifted_hist = Hist.new.Reg(100,-40,40,name="time",label="Time [ns]").Reg(100,200,800,name="waveform",label="Waveform [mV]").Double()
        waveform_filtered_shifted_hist.fill(ts_broadcast[select_peak_location,:].flatten()*1e9,waveforms_filtered_shifted[select_peak_location,:].flatten()*1e3)

        waveform_filtered_shifted_normalized_hist = Hist.new.Reg(100,-50,100,name="time",label="Time [ns]").Reg(100,-0.2,1.2,name="waveform",label="Waveform [arbitrary]").Double()
        waveform_filtered_shifted_normalized_hist.fill(ts_broadcast[select_peak_location,:].flatten()*1e9,waveforms_filtered_shifted_normalized[select_peak_location,:].flatten())

        amax_filtered_hist = Hist.new.Reg(110,250,800,name="peak_max",label=f"Peak Maximum [m{waveform_units}]").Double()
        amax_filtered_hist.fill(amax_filtered*1e3)
        #fit_gaus_pdf = fit_gaussians(amax_filtered*1e3,[(250,350),(350,450),(450,525),(550,615)])
        fit_e_height_pdf = fit_e_height(amax_filtered*1e3,4,limits=(250,615))

        fig, ax = mpl.subplots(figsize=(6,6),constrained_layout=True)
        amax_filtered_hist.plot(ax=ax,label="Data")
        #plot_pdf_over_hist(ax,fit_gaus_pdf,amax_filtered_hist,(250,615),label="Multi-Gaussian")
        plot_pdf_over_hist(ax,fit_e_height_pdf,amax_filtered_hist,(250,615),label="e$^-$ Height Fit")
        ax.set_title("Filtered Waveforms")
        ax.set_ylabel(f"Waveforms / {amax_filtered_hist.axes[0].widths[0]:.0f} mV")
        ax.legend()
        fig.savefig("max_hist.png")
        fig.savefig("max_hist.pdf")

        fig, ax = mpl.subplots(figsize=(6,6),constrained_layout=True)
        for i in range(min(nWaveforms,100)):
            #ax.plot(ts[:]*1e9,waveforms[i,:]*1e3,label="Unfiltered")
            ax.plot(ts[:]*1e9,waveforms_filtered[i,:]*1e3,label="filtered")
        ax.set_xlabel(f"Time [n{ts_units}]")
        ax.set_ylabel(f"Waveform [m{waveform_units}]")
        #ax.legend()
        fig.savefig("waveform.png")
        fig.savefig("waveform.pdf")

        fig, ax = mpl.subplots(figsize=(6,6),constrained_layout=True)
        ax.plot(fft_freqs,abs(waveform_fft_amp_mean),label="Unfiltered")
        ax.plot(fft_freqs,abs(waveform_fft_amp_mean*window_fft),label="Filtered")
        ax.set_xlabel(f"Frequency [{ts_units}$^{{-1}}$]")
        ax.set_ylabel(f"Waveform Amplitude [{waveform_units}]")
        ax.set_title(f"Average Spectrum Over {nWaveforms} Waveforms")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-2,1e3)
        ax.legend()
        fig.savefig("waveform_fft.png")
        fig.savefig("waveform_fft.pdf")

        fig, ax = mpl.subplots(figsize=(6,6),constrained_layout=True)
        ax.scatter([ts[x]*1e9 for x in argmax],amax*1e3,label="Unfiltered")
        ax.scatter(argmax_filtered_ts*1e9,amax_filtered*1e3,label="Filtered")
        ax.legend()
        ax.set_xlim(-10,40)
        ax.set_ylim(100,900)
        ax.set_xlabel(f"Peak Maximum Time [ns]")
        ax.set_ylabel(f"Peak Maximum [m{waveform_units}]")
        ax.set_title(f"Peak Max vs. Arg-max for {nWaveforms} Waveforms")
        fig.savefig("peak_maxVargmax.png")
        fig.savefig("peak_maxVargmax.pdf")

        fig, ax = mpl.subplots(figsize=(6,6),constrained_layout=False)
        waveform_filtered_hist.plot2d(ax=ax,norm=matplotlib.colors.PowerNorm(gamma=0.3,vmax=5000))
        ax.set_title("Filtered Waveforms")
        fig.savefig("waveform_hist.png")
        fig.savefig("waveform_hist.pdf")

        fig, ax = mpl.subplots(figsize=(6,6),constrained_layout=False)
        waveform_filtered_shifted_hist.plot2d(ax=ax,norm=matplotlib.colors.PowerNorm(gamma=0.3))#,vmax=5000))
        ax.set_title("Filtered, Peak-shifted Waveforms")
        fig.savefig("waveform_filtred_shifted_hist.png")
        fig.savefig("waveform_filtred_shifted_hist.pdf")

        fig, ax = mpl.subplots(figsize=(6,6),constrained_layout=False)
        waveform_filtered_shifted_normalized_hist.plot2d(ax=ax,norm=matplotlib.colors.PowerNorm(gamma=0.3))#,vmax=5000))
        ax.set_title("Filtered, Peak-shifted, Normalized Waveforms")
        fig.savefig("waveform_filtred_shifted_hist.png")
        fig.savefig("waveform_filtred_shifted_hist.pdf")

if __name__ == "__main__":
    import argparse
    ip = "192.168.55.2"

    parser = argparse.ArgumentParser(description='Collect and analyze pulses with an oscilloscope.')
    parser.add_argument("--analysisonly",'-a',
                        default=None,
                        help="Only perform analysis on the given file, don't collect data")
    parser.add_argument("--oscilloscope",'-o',
                        default=ip,
                        help=f"The location of the oscilloscope. Default: {ip}")
    parser.add_argument("--n_waveforms","-N",
                        type=int,
                        default=100,
                        help="Number of waveforms to collect. This is the number of waveforms written to file."
    )
    args = parser.parse_args()

    fn = args.analysisonly
    n_waveforms = args.n_waveforms
    if not fn:
        print("Collecting pulse waveform data...")
        fn = collect_pulser_waveform_data(ip,n_waveforms)
    else:
        print(f"Analyzing pulse waveform data from file: {fn}")
    analyze_pulses(fn)
