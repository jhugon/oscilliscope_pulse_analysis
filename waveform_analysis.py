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


def plot_hist_waveformVtime(waveforms_dset,nTBins=100,nVBins=100,tMin=None,tMax=None,vMin=None,vMax=None):
    waveform_units = waveforms_dset.attrs["units"]
    ts = waveforms_dset.dims[1][0]
    waveforms = waveforms_dset[:,:][np.amax(waveforms_dset,axis=1)!=0,:]
    waveforms = waveforms[np.amax(waveforms,axis=1) != np.amin(waveforms,axis=1),:]
    nWaveforms, waveform_len = waveforms.shape
    ts_broadcast, _ = np.broadcast_arrays(ts,waveforms)
    ts_units = ts.attrs["units"]

    if tMin is None:
        tMin = ts[0]
    if tMax is None:
        tMax = ts[-1]
    if vMin is None:
        vMin = min(waveforms[:,:].flatten())
    if vMax is None:
        vMax = max(waveforms[:,:].flatten())

    waveform_hist = Hist.new.Reg(nTBins,tMin,tMax,name="time",label=f"Time [{ts_units}]").Reg(nVBins,vMin,vMax,name="waveform",label=f"Waveform [{waveform_units}]").Double()
    waveform_hist.fill(ts_broadcast.flatten(),waveforms.flatten())

    fig, ax = mpl.subplots(figsize=(6,6),constrained_layout=False)
    waveform_hist.plot2d(ax=ax,norm=matplotlib.colors.PowerNorm(gamma=0.3))#,vmax=5000))
    return fig, ax

def plot_pdf_over_hist(ax,pdf,hist,limits,label="Fit"):
    x = np.linspace(limits[0],limits[1],1000)
    y = zfit.run(pdf.pdf(x))
    limited_hist = hist[bh.loc(limits[0]):bh.loc(limits[1])]
    y *= limited_hist[0:len:sum]
    bin_width = limited_hist.axes[0].widths[0]
    y *= bin_width
    ax.plot(x,y,label=label)
