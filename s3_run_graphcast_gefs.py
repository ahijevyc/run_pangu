import datetime
import numpy as np
import os
from pathlib import Path
import pdb
import sys
from typing import Any

import gc

import pandas as pd
import torch
import xarray as xr

from earth2studio.data import DataSource, GEFS_FX
from earth2studio.models.px import GraphCastOperational
from earth2studio.data.rx import LandSeaMask, SurfaceGeoPotential
from earth2studio.run import deterministic
from earth2studio.io import IOBackend, NetCDF4Backend
from earth2studio.utils.type import CoordSystem

class MemoryDataSource(DataSource):
    def __init__(self, data: xr.DataArray):
        super().__init__()
        self.data = data

    def __call__(self, init_time, variable, **kwargs):
        return self.data


# Define a custom IO class that subsets the data before writing to NetCDF.
class SubsetNetCDF4Backend(IOBackend):
    def __init__(
        self, file_name: str, lat_slice: slice, lon_slice: slice, backend_kwargs: dict = {}
    ):
        self.file_name = file_name
        self.lat_slice = lat_slice
        self.lon_slice = lon_slice
        self.backend_kwargs = backend_kwargs
        self.writer = None

    def add_array(
        self, coords: CoordSystem, array_name: str | list[str], **kwargs: dict[str, Any]
    ) -> None:
        # Create a temporary xarray object to correctly select coordinate values.
        # We don't need real data, just the coordinates and dimensions.
        dummy_data = np.zeros([len(v) for v in coords.values()])
        temp_da = xr.DataArray(dummy_data, coords=coords, dims=list(coords.keys()))

        # Select the subset using coordinate values (degrees)
        subset_da = temp_da.sel(lat=self.lat_slice, lon=self.lon_slice)

        # Extract the subsetted coordinates as a dictionary
        subset_coords = {k: v.values for k, v in subset_da.coords.items()}

        # Initialize the internal NetCDF4Backend with the subsetted coordinates
        self.writer = NetCDF4Backend(self.file_name, self.backend_kwargs)
        self.writer.add_array(subset_coords, array_name, **kwargs)

    def write(
        self,
        x: torch.Tensor | list[torch.Tensor],
        coords: CoordSystem,
        array_name: str | list[str],
    ) -> None:
        if self.writer is None:
            raise RuntimeError("add_array must be called before write.")

        if not isinstance(x, list):
            x = [x]
            array_name = [array_name]

        for i, tensor in enumerate(x):
            var_name = array_name[i]

            # 1. Create a DataArray from the incoming global data tensor
            temp_da = xr.DataArray(tensor.cpu().numpy(), coords=coords, dims=list(coords.keys()))

            # 2. Select the desired subset using coordinate values (degrees)
            subset_da = temp_da.sel(lat=self.lat_slice, lon=self.lon_slice)

            # 3. Extract the subsetted data and coordinates for the writer
            subset_x = torch.from_numpy(subset_da.data)
            subset_coords = {k: v.values for k, v in subset_da.coords.items()}

            # 4. Write the subsetted data using the internal writer
            self.writer.write(subset_x, subset_coords, var_name)

    def close(self):
        if self.writer:
            self.writer.close()

def run(init_time, model, members=["gec00"] + [f"gep{p:02d}" for p in range(1, 31)]):
    # The initialization time for the forecast
    # GEFS data is available at 00, 06, 12, 18 UTC.
    forecast_length = 240
    forecast_step_hours = 6
    nsteps = forecast_length // forecast_step_hours

    # Define the output directory for the ensemble forecasts
    output_dir = f"/glade/derecho/scratch/ahijevyc/ai-models/output/graphcast/{init_time:%Y%m%d%H}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Ensemble forecast outputs will be saved in: {output_dir}")
    # Get the grid the model expects by constructing it from its input coordinates
    lat = model.input_coords()["lat"]
    lon = model.input_coords()["lon"]
    static_data_dir = "/glade/derecho/scratch/ahijevyc/ai-models/static_data"
    os.makedirs(static_data_dir, exist_ok=True)
    z_filepath = os.path.join(static_data_dir, "graphcast_z_0.25deg.nc")
    lsm_filepath = os.path.join(static_data_dir, "graphcast_lsm_0.25deg.nc")
    static_data_time = datetime.datetime(2023, 1, 1)

    # Handle Geopotential (z)
    if os.path.exists(z_filepath):
        print(f"Loading z from local file: {z_filepath}")
        z = xr.open_dataarray(z_filepath)
    else:
        print(f"Fetching z and saving to: {z_filepath}")
        z_data = (
            SurfaceGeoPotential(cache=False)([static_data_time])
            .sel(lat=lat, lon=lon)
            .squeeze(dim="time")
        )
        z_data["variable"] = ["z"]
        z_data.to_netcdf(z_filepath)
        z = z_data

    # Handle Land-Sea Mask (lsm)
    if os.path.exists(lsm_filepath):
        print(f"Loading lsm from local file: {lsm_filepath}")
        lsm = xr.open_dataarray(lsm_filepath)
    else:
        print(f"Fetching lsm and saving to: {lsm_filepath}")
        lsm_data = (
            LandSeaMask(cache=False)([static_data_time]).sel(lat=lat, lon=lon).squeeze(dim="time")
        )
        lsm_data.to_netcdf(lsm_filepath)
        lsm = lsm_data

    print("Static data loaded successfully.")

    model_variables = model.input_coords()["variable"]
    vars_to_zero_fill = [v for v in model_variables if v.startswith("w") or v == "tp06"]

    vars_to_fetch = [v for v in model_variables if v not in vars_to_zero_fill]
    vars_to_fetch.remove("z")
    vars_to_fetch.remove("lsm")

    # 4. The Main Ensemble Loop
    for member in members:
        # Define the output path and the lat/lon slices
        output_filepath = os.path.join(output_dir, f"{member}.nc")
        lat_slice = slice(20, 60)
        lon_slice = slice(220, 300)

        # Check if a valid and complete output file already exists.
        if os.path.exists(output_filepath):
            try:
                # Use xarray to open the dataset. This is a more robust check for
                # a valid, closed NetCDF file than just using the netCDF4 library.
                with xr.open_dataset(output_filepath) as ds:
                    if len(ds.data_vars) != 85:
                        raise ValueError(
                            f"Incorrect # of data vars. Expected 85, found {len(ds.data_vars)}."
                        )
                    for dim_name, dim_size in ds.dims.items():
                        if dim_size == 0:
                            raise ValueError("Dim '{dim_name}' has size 0.")
                    if any(ds.z500.squeeze().max(dim=["lat", "lon"]) > 1e30):
                        raise ValueError(f"bad data in {output_filepath}")
                    print(f"Valid complete forecast exists for {init_time} '{member}', skipping.")
                    continue  # Skip to the next member
            except Exception as e:
                # This will catch errors if the file is corrupt, not a valid NetCDF,
                # or failed our completeness check.
                print(
                    f"Found invalid or incomplete file for member '{member}', removing. Error: {e}"
                )
                os.remove(output_filepath)

        # 4a. Fetch initial conditions for the current member
        print(f"Fetching initial conditions for {member} at {init_time.isoformat()}...")
        gefs_source = GEFS_FX(member=member)
        initial_state_partial = gefs_source(init_time, [datetime.timedelta(hours=0)], vars_to_fetch)

        # Manually regrid the data to the target grid
        print(f"Regridding initial state for {member}...")
        wrapped = initial_state_partial.sel(lon=0).assign_coords(lon=360)
        initial_state_periodic = xr.concat([initial_state_partial, wrapped], dim="lon")
        initial_state_partial = initial_state_periodic.interp(lat=lat, lon=lon, method="linear")

        data_arrays_to_concat = [initial_state_partial]
        for var_name in vars_to_zero_fill:
            zero_array = xr.zeros_like(initial_state_partial.isel(variable=0))
            zero_array["variable"] = var_name
            data_arrays_to_concat.append(zero_array)
        data_arrays_to_concat.extend([z, lsm])
        initial_state = xr.concat(data_arrays_to_concat, dim="variable", coords="minimal")
        assert initial_state.notnull().all()
        # The 'lead_time' dimension must be removed for the MemoryDataSource
        initial_state = initial_state.sel(variable=model_variables).squeeze(
            dim="lead_time", drop=True
        )

        # 4b. Run the forecast using the fetched initial state
        in_memory_source = MemoryDataSource(initial_state)

        # Instantiate our custom writer
        subset_writer = SubsetNetCDF4Backend(
            file_name=output_filepath,
            lat_slice=lat_slice,
            lon_slice=lon_slice,
            backend_kwargs={"mode": "w"},
        )

        # Run the deterministic forecast
        deterministic([init_time], nsteps, model, in_memory_source, subset_writer)

        # Close the writer to finalize the file on disk
        subset_writer.close()

        print(f"Successfully created subset forecast file: {output_filepath}")
        print(f"--- Finished forecast for member '{member}' ---")
        del initial_state
        del in_memory_source
        del subset_writer
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n✅ All ensemble member forecasts for {init_time} been successfully generated.")


print("Initializing GraphCast model...")
model_class = GraphCastOperational
model = model_class.load_model(model_class.load_default_package())
print("Model initialized successfully.")

if __name__ == '__main__':
    init_time = pd.to_datetime(sys.argv[1])
    run(init_time, model)
