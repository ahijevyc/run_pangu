import numpy as np
import os
import pandas as pd
import pdb
import xarray as xr

from earth2studio.models.px import GraphCastOperational
from earth2studio.data import DataArrayFile, GFS
from earth2studio.data.base import DataSource
from earth2studio.data.rx import LandSeaMask, SurfaceGeoPotential
from earth2studio.io import NetCDF4Backend
from earth2studio.run import deterministic

model = GraphCastOperational.load_model(GraphCastOperational.load_default_package())


class GFSFill(GFS):
    """
    Intercepts requests for specified variables and provides predefined data arrays.
    For all other variables, it falls back to the standard GFS implementation.
    """

    def __init__(self, custom_arrays: dict[str, xr.DataArray], *args, **kwargs):
        """
        Initializes with a dictionary of custom xarray.DataArrays,
        where the keys are the variable names.
        """
        super().__init__(*args, **kwargs)
        self.custom_arrays = custom_arrays
        self.custom_vars = list(custom_arrays.keys())

    def __call__(
        self,
        time: np.ndarray,
        variable: np.ndarray,
    ) -> xr.DataArray:
        
        # Identify which custom variables are in the current request
        requested_custom_vars = [v for v in self.custom_vars if v in variable]
        
        # Identify which variables need to be fetched from the standard GFS source
        other_vars = [v for v in variable if v not in self.custom_vars]

        # Fetch data for non-custom variables from the parent GFS class
        if other_vars:
            gfs_data = super().__call__(time, np.array(other_vars))
        else:
            gfs_data = None

        # Prepare a list of all data arrays to be concatenated
        data_to_concat = []
        if gfs_data is not None:
            data_to_concat.append(gfs_data)
        
        # Add the requested custom arrays to the list
        for var_name in requested_custom_vars:
            # Expand dims to match the structure of the fetched data
            custom_da = self.custom_arrays[var_name].expand_dims({"time": time})
            data_to_concat.append(custom_da)

        # Concatenate all data arrays along the 'variable' dimension
        if len(data_to_concat) > 1:
            return xr.concat(data_to_concat, dim="variable")
        elif len(data_to_concat) == 1:
            return data_to_concat[0]
        else:
            # Should not happen if 'variable' is never empty
            return xr.DataArray()

ds = GFS()
pdb.set_trace()
# cache=False to avoid AttributeError: type object 'WholeFileCacheFileSystem' has no attribute '_cat_file'. Did you mean: 'cat_file'?
# dummy time list for required positional argument 'time'
# squeeze 'time' to avoid ValueError: Dimension time already exists.
zsl = SurfaceGeoPotential(cache=False)([0]).squeeze(dim="time")
lsm = LandSeaMask(cache=False)([0]).squeeze(dim="time")

# --- Instantiate Custom Data Source ---
custom_data = {"zsl": zsl, "lsm": lsm}
ds_fill = GFSFill(custom_arrays=custom_data)

# --- Run Forecast ---
itime = pd.to_datetime("2024-05-01")
nsteps=8
io = NetCDF4Backend("GFS.nc", backend_kwargs={'mode': 'w'})
deterministic([itime], nsteps, model, ds, io)
io = NetCDF4Backend("GFSFill.nc", backend_kwargs={'mode': 'w'})
deterministic([itime], nsteps, model, ds_fill, io)

print("Forecast run complete.")

