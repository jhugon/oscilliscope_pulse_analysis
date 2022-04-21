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


def collect_pulser_waveform_data(ip,nWaveforms,keep_settings=False):
    channel="channel1"
    trigger_threshold =200e-3
    vert_scale = 200e-3
    vert_offset = -400e-3
    horiz_scale = 50e-9
    horiz_offset = 0
    if not keep_settings:
        setup_vert(ip,vert_scale,vert_offset,probe=1,bwlimit="20M",channel=channel)
        setup_horiz(ip,horiz_scale,horiz_offset)
        setup_trig(ip,trigger_threshold,10e-6,sweep="single",channel=channel)

    print(f"Collecting {nWaveforms} waveforms")
    print(f"Ensure oscilloscope {channel} is hooked up to the the DUT output.")
    description = get_description()

    now = datetime.datetime.now().replace(microsecond=0)
    out_file_name = "pulses_{}_{:d}waveforms.hdf5".format(now.isoformat(),nWaveforms)
    print(f"Output filename is: {out_file_name}")
    with h5py.File(out_file_name,"w") as out_file:
        out_file.attrs["description"] = description
        out_file.attrs["starttime"] = now.isoformat()
        out_file.attrs["status"] = "fail"
        collect_waveforms(ip,out_file,nWaveforms,channel=channel)
        out_file.attrs["status"] = "success"
    return out_file_name

def analyze_multiple_pulse_files(fns):

    fig_fft, ax_fft = mpl.subplots(figsize=(6,6),constrained_layout=True)
    ax_fft.set_xlabel(f"Frequency [Hz]")
    ax_fft.set_ylabel(f"Waveform Amplitude [V]")
    ax_fft.set_title(f"Average Spectrum Over Waveforms")
    ax_fft.set_xscale("log")
    ax_fft.set_yscale("log")

    fig_vmin, ax_vmin = mpl.subplots(figsize=(6,6),constrained_layout=True)
    ax_vmin.set_title(f"Waveform Minimum")
    ax_vmin.set_yscale("log")
    fig_vmax, ax_vmax = mpl.subplots(figsize=(6,6),constrained_layout=True)
    ax_vmax.set_title(f"Waveform Maximum")
    ax_vmax.set_yscale("log")

    for fn in fns:
        with h5py.File(fn) as in_file:
            run_description = in_file.attrs["description"]
            run_starttime = in_file.attrs["starttime"]
            run_status = in_file.attrs["status"]
            print(f"Run file: {fn}")
            print(f"Run start time: {run_starttime}")
            print(f"Run description: {run_description}")
            if run_status != "success":
                print(f"Run status: failure, skipping file")
                continue
            waveforms_dset = None
            try:
                waveforms_dset = in_file["waveforms_raw"]
            except KeyError:
                print(f"Error retreiving \"waveforms_raw\" dataset from file {fn}. Root of file contains: {list(in_file.keys())}",file=sys.stderr)
                print("Exiting.",file=sys.stderr)
                sys.exit(1)
            waveform_units = "V"
            ts = waveforms_dset.dims[1][0]
            waveforms = calibrate_waveforms(waveforms_dset)
            nWaveforms, waveform_len = waveforms.shape
            ts_broadcast, _ = np.broadcast_arrays(ts,waveforms)
            ts_units = ts.attrs["units"]
            sample_period = ts[1]-ts[0]
            sample_frequency = 1./sample_period

            amax = np.amax(waveforms,axis=1)
            argmax = np.argmax(waveforms,axis=1)
            amin = np.amin(waveforms,axis=1)
            argmin = np.argmin(waveforms,axis=1)
            argmax_ts = np.array([ts[x] for x in argmax])

            waveform_ffts = fft.rfft(waveforms[:,:])
            fft_freqs = fft.rfftfreq(waveforms.shape[-1],d=sample_period)
            waveform_fft_amp_mean = np.sum(abs(waveform_ffts),axis=0)/nWaveforms
            ax_fft.plot(fft_freqs,abs(waveform_fft_amp_mean),label=run_description)

            waveform_hist = make_hist_waveformVtime(waveforms_dset,time_units="ns",voltage_units="mV",downsample_time_by=10)
            fig, ax = mpl.subplots(figsize=(6,6),constrained_layout=False)
            fig.suptitle(f"{run_starttime}\n{run_description}",wrap=True)
            waveform_hist.plot2d(ax=ax,norm=PHOSPHOR_HIST_NORM)
            fig.savefig(f"waveform_hist_{run_starttime}.png")
            mpl.close(fig)

            waveform_min_hist, waveform_max_hist = make_hist_waveformMinMax(waveforms_dset,voltage_units="mV")
            waveform_min_hist.plot(ax=ax_vmin,label=run_description)
            waveform_max_hist.plot(ax=ax_vmax,label=run_description)

    ax_fft.legend()
    fig_fft.savefig("waveform_fft.png")
    fig_fft.savefig("waveform_fft.pdf")

    ax_vmin.legend()
    ax_vmax.legend()
    fig_vmin.savefig("waveform_vmin.png")
    fig_vmin.savefig("waveform_vmin.pdf")
    fig_vmax.savefig("waveform_vmax.png")
    fig_vmax.savefig("waveform_vmax.pdf")


if __name__ == "__main__":
    import argparse
    ip = "192.168.55.2"

    n_waveforms_defualt = 100

    parser = argparse.ArgumentParser(description='Collect and analyze pulses with an oscilloscope.')
    parser.add_argument("--printmetadata",'-p',
                        default=None,
                        help="Only print file metadata. Only one path allowed, but you may glob within quotes like './analyze_pulses.py -p \"pulses*.hdf5\"'")
    parser.add_argument("--analysisonly",'-a',
                        default=None,
                        help="Only perform analysis on the given file, don't collect data")
    parser.add_argument("--oscilloscope",'-o',
                        default=ip,
                        help=f"The location of the oscilloscope. Default: {ip}")
    parser.add_argument("--n_waveforms","-N",
                        type=int,
                        default=n_waveforms_defualt,
                        help=f"Number of waveforms to collect. This is the number of waveforms written to file. Default: {n_waveforms_defualt}"
    )
    parser.add_argument("--keepsettings","-k",
                        action="store_true", default=False,
                        help=f"Keep current oscilloscope vertical, horizontal, and trigger settings isntead of changing to the standardized settings."
    )
    args = parser.parse_args()

    fn = args.analysisonly
    ip = args.oscilloscope
    n_waveforms = args.n_waveforms
    if args.printmetadata:
        print("Pulse waveform file metadata:")
        print_metadata(args.printmetadata,"/")
        sys.exit(0)
    if not fn:
        print("Collecting pulse waveform data...")
        fn = collect_pulser_waveform_data(ip,n_waveforms,keep_settings=args.keepsettings)
    else:
        print(f"Analyzing pulse waveform data from file(s): {fn}")
    #analyze_pulses(fn)
    fns = fn.split(" ")
    fns = [glob.glob(fn) for fn in fns]
    fns = [fn for sublist in fns for fn in sublist]
    analyze_multiple_pulse_files(fns)
