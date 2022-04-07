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

def calibrate_waveforms(waveforms_dset):
    slope = waveforms_dset.attrs["calib_slope"]
    intercept = waveforms_dset.attrs["calib_intercept"]
    return waveforms_dset[:,:]*slope+intercept

def decode_time_units(s):
    """
    Decodes units string like "s", "us", "ns" etc.

    returns float scale factor and latex string
    """

    if s == "s":
        return 1., "s"
    elif s == "ms":
        return 1.e3, "ms"
    elif s == "us":
        return 1.e6, "μs"
    elif s == "ns":
        return 1.e9, "ns"
    elif s == "ps":
        return 1.e12, "ps"
    elif s == "fs":
        return 1.e15, "fs"
    else:
        raise ValueError(f"Couldn't decode time unit: '{s}'")

def decode_voltage_units(s):
    """
    Decodes units string like "V", "v", "mV", "mv", "uV" etc.

    returns float scale factor and latex string
    """

    if s.lower() == "kv":
        return 1.e-3, "kV"
    elif s.lower() == "v":
        return 1., "V"
    elif s.lower() == "mv":
        return 1.e3, "mV"
    elif s.lower() == "uv":
        return 1.e6, "μV"
    elif s.lower() == "nv":
        return 1.e9, "nV"
    else:
        raise ValueError(f"Couldn't decode voltage unit: '{s}'")

def plot_hist_waveformVtime(ax,waveforms_dset,time_units="us",voltage_units="V",downsample_time_by=100):
    """
    Plot a "phosphor oscilloscope" looking histogram of the waveforms in waveforms_dset on matplotlib ax.

    time_units: s, ms, us, ns

    v_units: V, mV, uV

    downsample_time_by: reduce the binning of time by this factor (can be int or float)
    """
    ts = waveforms_dset.dims[1][0]
    nWaveforms, waveform_len = waveforms_dset.shape
    ts_broadcast, _ = np.broadcast_arrays(ts,waveforms_dset[:,:])

    time_sf, time_units = decode_time_units(time_units)
    voltage_sf, voltage_units = decode_voltage_units(voltage_units)

    nTBins = waveform_len+1
    nTBins = nTBins // downsample_time_by
    tMin = ts[0]*time_sf
    tMax = (ts[-1]+ts.attrs["sample_spacing"])*time_sf

    nVBins = waveforms_dset.attrs["calib_N_steps"]+1
    vMin = waveforms_dset.attrs["calib_min"]*voltage_sf
    vMax = (waveforms_dset.attrs["calib_max"]+waveforms_dset.attrs["calib_slope"])*voltage_sf

    waveforms = calibrate_waveforms(waveforms_dset)

    waveform_hist = Hist.new.Reg(nTBins,tMin,tMax,name="time",label=f"Time [{time_units}]").Reg(nVBins,vMin,vMax,name="waveform",label=f"Waveform [{voltage_units}]").Double()
    waveform_hist.fill(ts_broadcast.flatten()*time_sf,waveforms.flatten()*voltage_sf)

    waveform_hist.plot2d(ax=ax,norm=matplotlib.colors.PowerNorm(gamma=0.3))#,vmax=5000))

def plot_pdf_over_hist(ax,pdf,hist,limits,label="Fit"):
    x = np.linspace(limits[0],limits[1],1000)
    y = zfit.run(pdf.pdf(x))
    limited_hist = hist[bh.loc(limits[0]):bh.loc(limits[1])]
    y *= limited_hist[0:len:sum]
    bin_width = limited_hist.axes[0].widths[0]
    y *= bin_width
    ax.plot(x,y,label=label)

if __name__ == "__main__":
    fn = "dummy_waveforms_2022-04-07T09:41:47_20waveforms.hdf5"
    with h5py.File(fn) as f:
        waveforms_raw = f["waveforms_raw"]
        fig, ax = mpl.subplots(figsize=(6,6))
        plot_hist_waveformVtime(ax,waveforms_raw,voltage_units="mV")
        fig.savefig("test.png")
