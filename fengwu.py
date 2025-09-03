#!/usr/bin/env python
# coding: utf-8

# works with conda env:pangu
# and with conda env:ainwp as constructed in README
# run with 1 GPU on vt100 (gp100 doesn't have enough VRAM; needs 17+GB)
# and had to save new default casper modules (lower ncarenv version than default)
# Currently Loaded Modules:
#   1) ncarenv/23.10 (S)   2) conda/latest   3) cuda/11.8.0   4) cudnn/8.7.0.84-11.8
# 
#   Where:
#    S:  Module is Sticky, requires --force to unload or purge
# 
# Inferences in conda env:ainwp is different by 0.0001 K from pangu env (after 240 hours)
# conda env:ainwp is supposed to replicate realtime runs, but still 0.1 K different from realtime runs

# In[1]:


import argparse
import os
from pathlib import Path
import sys

import earthkit.data as ekd
import numpy as np
import pandas as pd
import xarray
from run_pangu import plot_ensemble
from run_pangu.s3_run_fengwu_ecmwf import (
    fengwu_channels,
    channel_subset,
    lat,
    lon,
    pressure_levels,
    setup_model_sessions,
    variables,
)

ai_models_dir = Path("/glade/derecho/scratch/ahijevyc/ai-models")
date = pd.to_datetime(sys.argv[1])
date_6 = date - pd.to_timedelta("6h")
ic = "gefs"


# In[2]:


def fengwu_input(grb):
    input = ekd.from_source("file", grb)
    pressure_levels_int = [int(p) for p in pressure_levels]
    sfc_param = ["10u", "10v", "2t", "msl"]
    fields_sfc = input.sel(param=sfc_param, levtype="sfc").order_by(param=sfc_param)
    fields_pl = input.sel(param=variables, level=pressure_levels_int, levtype="pl").order_by(
        param=variables, level=pressure_levels_int
    )
    fields_all = fields_sfc + fields_pl
    # print(fields_all.ls())
    return fields_all.to_numpy()


def fengwu_input_nc(nc):
    input = xarray.open_dataset(nc)
    sfc_param = ["u10m", "v10m", "t2m", "msl"]
    pl_param = [f"{f}{p}" for f in variables for p in pressure_levels]
    fields_all = []
    for p in sfc_param + pl_param:
        field = input["__xarray_dataarray_variable__"].sel(channel=p).squeeze().values
        fields_all.append(field)
    return np.stack(fields_all)


def run_fengwu(input, data_mean, data_std, date, ic, odir, clobber=False):
    for fhr in range(6, 246, 6):
        output_filename = f"{odir}/fengwu_{ic}_pred_{fhr:03d}.nc"
        if os.path.exists(output_filename):
            if clobber:
                os.remove(output_filename)
            else:
                continue
        print(f"Processing {date:%Y-%m-%d} - {fhr} hour")
        # 
        output = ort_session_6.run(None, {"input": input})[0]
        input = np.concatenate((input[:, 69:], output[:, :69]), axis=1)
        output = (output[0, :69] * data_std) + data_mean

        # Create prediction timedelta
        pred_timedelta = pd.Timedelta(hours=fhr)

        # Create xarray DataArrays with proper dimensions
        da_output = xarray.DataArray(
            data=np.expand_dims(np.expand_dims(output, axis=0), axis=0),
            coords={
                "init_time": [date],
                "prediction_timedelta": [pred_timedelta],
                "channel": fengwu_channels,
                "lat": lat,
                "lon": lon,
            },
            dims=["init_time", "prediction_timedelta", "channel", "lat", "lon"],
        ).sel(lat=slice(60, 20), lon=slice(220, 300), channel=channel_subset)

        # Save as netCDF
        da_output.to_netcdf(output_filename)


model_dir = Path("/glade/derecho/scratch/zxhua/AI_global_forecast_model_for_education/FengWu/model")
ort_session_6 = setup_model_sessions(model_dir)
# Load normalization data
data_mean = np.load(model_dir / "data_mean.npy")[:, np.newaxis, np.newaxis]
data_std = np.load(model_dir / "data_std.npy")[:, np.newaxis, np.newaxis]



for mem in ["c00"] + [f"p{m:02d}" for m in range(1, 31)]:  # gefs has 30 members; ecmwf has 50
    ens = int(mem[1:])  # int() removes leading zeros
    assert mem.startswith("p") or mem == "c00"
    odir = ai_models_dir / f"output/fengwu/{date:%Y%m%d%H}/{mem}"
    if all([Path(f"{odir}/fengwu_{ic}_pred_{fhr:03d}.nc").exists() for fhr in range(6,246,6)]):
        print(f"{date} {mem} is complete")
        continue

    if ic == "ecmwf":
        assert date > pd.to_datetime("20250209"), "started saving ecmwf after 20250209"
        prior_nc = f"/glade/derecho/scratch/sobash/fengwu_realtime/{date:%Y%m%d%H}/ens{ens}/pangu_ens{ens}_init_{date_6:%Y%m%d%H}.nc"
        current_nc = f"/glade/derecho/scratch/sobash/fengwu_realtime/{date:%Y%m%d%H}/ens{ens}/pangu_ens{ens}_init_{date:%Y%m%d%H}.nc"
        input_prior = fengwu_input_nc(prior_nc)
        input_current = fengwu_input_nc(current_nc)
    elif ic == "gefs":
        from run_pangu import s1_get_gefs, s2_make_ic_gefs

        assert ens <= 30
        s1_get_gefs.download_time(date_6)
        s1_get_gefs.download_time(date)

        s2_make_ic_gefs.process_member(date_6.strftime("%Y%m%d%H"), mem)
        s2_make_ic_gefs.process_member(date.strftime("%Y%m%d%H"), mem)
        prior_grb = (
            ai_models_dir / f"input/{date_6:%Y%m%d%H}/{mem}/ge{mem}.t{date_6:%H}z.pgrb.0p25.f000"
        )
        current_grb = (
            ai_models_dir / f"input/{date:%Y%m%d%H}/{mem}/ge{mem}.t{date:%H}z.pgrb.0p25.f000"
        )
        input_prior = fengwu_input(prior_grb)
        input_current = fengwu_input(current_grb)

    # Normalize input data
    input_current_after_norm = (input_current - data_mean) / data_std
    input_prior_after_norm = (input_prior - data_mean) / data_std
    input_fengwu = np.concatenate((input_prior_after_norm, input_current_after_norm), axis=0)[
        np.newaxis, :, :, :
    ]
    input_fengwu = input_fengwu.astype(np.float32)

    os.makedirs(odir, exist_ok=True)
    run_fengwu(input_fengwu, data_mean, data_std, date, ic, odir)
