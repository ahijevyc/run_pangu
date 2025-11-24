"""
Perhaps use instead
from earth2studio.utils.time import xtime
"""

import logging
from pathlib import Path
import pandas as pd
import xarray


def xtime(ds: xarray.Dataset):
    """convert xtime variable to datetime and assign to coordinate"""

    # remove one-element-long Time dimension
    ds = ds.squeeze(dim="Time", drop=True)

    logging.info("decode initialization time variable")
    initial_time = pd.to_datetime(
        ds["initial_time"].load().item().decode("utf-8").strip(),
        format="%Y-%m-%d_%H:%M:%S",
    )

    # assign initialization time variable to its own coordinate
    ds = ds.assign_coords(
        initial_time=(
            ["initial_time"],
            [initial_time],
        ),
    )

    # extract member number from part of file path
    # assign to its own coordinate
    filename = Path(ds.encoding["source"])
    mem = [p for p in filename.parts if p.startswith("mem")]
    if mem:
        mem = mem[0].lstrip("mem_")
        mem = int(mem)
        ds = ds.assign_coords(mem=(["mem"], [mem]))

    logging.info("decode valid time and assign to variable")
    valid_time = pd.to_datetime(
        ds["xtime"].load().item().decode("utf-8").strip(),
        format="%Y-%m-%d_%H:%M:%S",
    )
    ds = ds.assign(valid_time=[valid_time])

    # calculate forecast hour and assign to variable
    forecastHour = (valid_time - initial_time) / pd.to_timedelta(1, unit="hour")
    ds = ds.assign(forecastHour=float(forecastHour))

    return ds
