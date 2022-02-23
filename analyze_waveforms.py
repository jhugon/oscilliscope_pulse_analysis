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


def fit_e_height(data_np,nPeaks,limits):
    obs = zfit.Space("peak_max_for_e_height",limits=limits)
    e_height = zfit.Parameter(f"e_height",95,50,150)
    delta_mus = []
    mus = []
    yields = []
    sigmas = []
    base_pdfs = []
    ext_pdfs = []
    #constraints = [zfit.constraint.GaussianConstraint(params=e_height,observation=95.,uncertainty=10.)]
    constraints = []
    peak_nums = []
    for i in range(3,3+nPeaks):
        peak_num = zfit.Parameter(f"peak_num_{i}",i,floating=False)
        delta_mu = zfit.Parameter(f"delta_m_{i}", 0, -20,20)
        mu = zfit.ComposedParameter(f"m_{i}", lambda eh, dm,pn: (pn)*eh+dm,params=[e_height,delta_mu,peak_num])
        sigma = zfit.Parameter(f"sig_{i}", 16.,10,25)
        yield_gauss = zfit.Parameter(f"yld_{i}",300,200,400)
        gauss = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)
        delta_mus.append(delta_mu)
        mus.append(mu)
        sigmas.append(sigma)
        yields.append(yield_gauss)
        base_pdfs.append(gauss)
        ext_pdfs.append(gauss.create_extended(yield_gauss))
        constraints.append(zfit.constraint.GaussianConstraint(params=delta_mu,observation=0.,uncertainty=20.))
    sum_pdf = zfit.pdf.SumPDF(pdfs=ext_pdfs)

    data = zfit.Data.from_numpy(obs=obs,array=data_np)
    nll = zfit.loss.ExtendedUnbinnedNLL(model=sum_pdf,data=data,constraints=constraints)

    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(nll)
    param_uncert = result.hesse()
    print(result)

    return sum_pdf

def fit_gaussians(data_np,limits_list):
    nLimits = len(limits_list)
    obs = zfit.Space("peak_max",limits=(limits_list[0][0],limits_list[-1][-1]))
    mus = []
    yields = []
    sigmas = []
    base_pdfs = []
    ext_pdfs = []
    for i, limits in enumerate(limits_list):
        mu = zfit.Parameter(f"mu_{i}", 0.5*(limits[1]+limits[0]),limits[0],limits[1])
        sigma = zfit.Parameter(f"sigma_{i}", 50,10,100)
        yield_gauss = zfit.Parameter(f"yield_{i}",1e2,0,1e3)
        gauss = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)
        mus.append(mu)
        sigmas.append(sigma)
        yields.append(yield_gauss)
        base_pdfs.append(gauss)
        ext_pdfs.append(gauss.create_extended(yield_gauss))
    sum_pdf = zfit.pdf.SumPDF(pdfs=ext_pdfs)

    data = zfit.Data.from_numpy(obs=obs,array=data_np)
    nll = zfit.loss.ExtendedUnbinnedNLL(model=sum_pdf,data=data)

    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(nll)
    param_uncert = result.hesse()
    print(result)

    return sum_pdf

def plot_pdf_over_hist(ax,pdf,hist,limits,label="Fit"):
    x = np.linspace(limits[0],limits[1],1000)
    y = zfit.run(pdf.pdf(x))
    limited_hist = hist[bh.loc(limits[0]):bh.loc(limits[1])]
    y *= limited_hist[0:len:sum]
    bin_width = limited_hist.axes[0].widths[0]
    y *= bin_width
    ax.plot(x,y,label=label)

freq_cutoff = 200e6 # Hz

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
