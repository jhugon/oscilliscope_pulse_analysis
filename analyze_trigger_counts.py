#!/usr/bin/env python3

from oscilloscope import *
import time
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as mpl
import datetime
import h5py

import vxi11

def normal_counts(ip):
    now = datetime.datetime.now().replace(microsecond=0)
    setup_vert(ip,200e-3,-400e-3,probe=1,bwlimit="20M")
    setup_horiz(ip,100e-9,0)
    setup_trig(ip,100e-3,10e-6)
    time_per_trig_val = 1
    trig_max = 850
    trig_min = 350
    trig_n_vals = 20
    trigger_values = np.linspace(trig_min,trig_max,trig_n_vals)
    print(f"Spending {time_per_trig_val} s triggering on each of {trig_n_vals} values between {trig_min} and {trig_max} mV")
    out_file_name = "counts_{}_{:d}trigs_{:.0f}to{:.0f}mV_{:.0f}s.hdf5".format(now.isoformat(),trig_n_vals,trig_min,trig_max,time_per_trig_val)
    print(f"Output filename is: {out_file_name}")
    with h5py.File(out_file_name,"w") as out_file:
        collect_counter_data(ip,out_file,trigger_values,time_per_trig_val)
    return out_file_name

def max_resolution_counts(ip):
    """
    trigger must be a multiple of 8 mV
    """
    now = datetime.datetime.now().replace(microsecond=0)
    setup_vert(ip,200e-3,-400e-3,probe=1,bwlimit="20M")
    setup_horiz(ip,100e-9,0)
    setup_trig(ip,100e-3,10e-6)
    time_per_trig_val = 1
    trigger_values = np.arange(344,850,8)
    trig_max = trigger_values[-1]
    trig_min = trigger_values[0]
    trig_n_vals = len(trigger_values)
    print(f"Spending {time_per_trig_val} s triggering on each of {trig_n_vals} values between {trig_min} and {trig_max} mV")
    out_file_name = "counts_max_res_{}_{:d}trigs_{:.0f}to{:.0f}mV_{:.0f}s.hdf5".format(now.isoformat(),trig_n_vals,trig_min,trig_max,time_per_trig_val)
    print(f"Output filename is: {out_file_name}")
    with h5py.File(out_file_name,"w") as out_file:
        collect_counter_data(ip,out_file,trigger_values,time_per_trig_val)
    return out_file_name

def analyze_counts(in_file_name):
    with h5py.File(in_file_name) as in_file:
        counts = in_file["counts"]
        trigger_values = in_file["trigger_values"]
        trigger_units = trigger_values.attrs["units"]
        count_errors = np.sqrt(counts)

        #x = np.linspace(trigger_values[0],trigger_values[-1],1000)
        x = np.linspace(80,350,1000)
        x = np.linspace(350,750,1000)
        spline = UnivariateSpline(trigger_values[:],counts[:],s=len(counts))
        spline_deriv = spline.derivative()

        brackets=[(115,140),(250,260)]
        brackets=[]
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

if __name__ == "__main__":
    ip = "192.168.55.2"
    #fn = normal_counts(ip)
    fn = max_resolution_counts(ip)
    analyze_counts(fn)
