#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as mpl
import h5py

#in_file_name = "counts_2022-02-14T15:43:17_51trigs_0to500mV_3s.hdf5"
in_file_name = "counts_max_res_2022-02-14T16:47:46_29trigs_80to304mV_1s.hdf5"
with h5py.File(in_file_name) as in_file:
    counts = in_file["counts"]
    trigger_values = in_file["trigger_values"]
    trigger_units = trigger_values.attrs["units"]
    count_errors = np.sqrt(counts)

    #x = np.linspace(trigger_values[0],trigger_values[-1],1000)
    x = np.linspace(80,350,1000)
    spline = UnivariateSpline(trigger_values[:],counts[:],s=len(counts))
    spline_deriv = spline.derivative()

    brackets=[(115,140),(250,260)]
    peaks = []
    for bracket in brackets:
        res = minimize_scalar(spline_deriv,bracket)
        if res.success:
            peaks.append(res.x)
    print("Derivative peaks found:")
    for p in peaks:
        print(f"  {p:.2f} {trigger_units}")
    if len(peaks) == 2:
        print(f"Difference in peaks: {peaks[1]-peaks[0]:.2f} {trigger_units}")

    fig, ax = mpl.subplots(figsize=(6,6),constrained_layout=True)
    ax.errorbar(trigger_values[:],counts[:],yerr=count_errors,fmt=".",label="Data")
    ax.plot(x,spline(x),"-g",label="Spline")
    for p in peaks:
        ax.axvline(p,c="r")
    ax.set_xlabel(f"Trigger Threshold [{trigger_units}]")
    ax.set_ylabel(f"Triggers / {counts.attrs['time_interval']} {counts.attrs['time_interval_units']}")
    ax.set_title("SiPM Trigger Rate")
    #fig.savefig("trigger_rate.png")
    ax.set_yscale("log")
    fig.savefig("trigger_rate_log.png")

    fig, ax = mpl.subplots(figsize=(6,6),constrained_layout=True)
    ax.plot(x,-spline_deriv(x),"-g")
    for p in peaks:
        ax.axvline(p,c="r")
    ax.set_xlabel(f"Trigger Threshold [{trigger_units}]")
    ax.set_ylabel(f"-Derivative of Triggers / {counts.attrs['time_interval']} {counts.attrs['time_interval_units']}")
    ax.set_title("-Derivative of SiPM Trigger Rate")
    #fig.savefig("trigger_rate_deriv.png")
    ax.set_yscale("log")
    fig.savefig("trigger_rate_deriv_log.png")
