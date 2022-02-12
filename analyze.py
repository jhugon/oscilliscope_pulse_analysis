#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as mpl
import h5py

in_file_name = "raw_2022-02-12T16:56:16.hdf5"
with h5py.File(in_file_name) as in_file:
    counts = in_file["counts"]
    trigger_values = in_file["trigger_values"]
    trigger_units = trigger_values.attrs["units"]
    count_errors = np.sqrt(counts)

    fig, ax = mpl.subplots(figsize=(6,6),constrained_layout=True)
    ax.errorbar(trigger_values[:],counts[:],yerr=count_errors,fmt=".")
    ax.set_xlabel(f"Trigger Threshold [{trigger_units}]")
    ax.set_ylabel(f"Triggers / {counts.attrs['time_interval']} {counts.attrs['time_interval_units']}")
    ax.set_title("SiPM Trigger Rate")
    fig.savefig("trigger_rate.png")
    ax.set_yscale("log")
    fig.savefig("trigger_rate_log.png")

    counts_added = np.add.accumulate(counts)
    fig, ax = mpl.subplots(figsize=(6,6),constrained_layout=True)
    ax.errorbar(trigger_values[:],counts_added[:],yerr=np.sqrt(counts_added),fmt=".")
    ax.set_xlabel(f"Trigger Threshold [{trigger_units}]")
    ax.set_ylabel(f"Triggers < X / {counts.attrs['time_interval']} {counts.attrs['time_interval_units']}")
    ax.set_title("SiPM Trigger Rate Integrated")
    fig.savefig("trigger_rate_int.png")
    ax.set_yscale("log")
    fig.savefig("trigger_rate_int_log.png")

    counts_added_flipped = counts[:].sum()-counts_added
    fig, ax = mpl.subplots(figsize=(6,6),constrained_layout=True)
    ax.errorbar(trigger_values[:],counts_added_flipped[:],yerr=np.sqrt(counts_added_flipped),fmt=".")
    ax.set_xlabel(f"Trigger Threshold [{trigger_units}]")
    ax.set_ylabel(f"Triggers > X / {counts.attrs['time_interval']} {counts.attrs['time_interval_units']}")
    ax.set_title("SiPM Trigger Rate Integrated")
    fig.savefig("trigger_rate_int_flipped.png")
    ax.set_yscale("log")
    fig.savefig("trigger_rate_int_flipped_log.png")
