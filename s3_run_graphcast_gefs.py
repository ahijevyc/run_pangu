import datetime
import numpy as np
import os
import pdb
import sys

import pandas as pd
import xarray as xr

from earth2studio.data import DataSource
from earth2studio.data import GEFS_FX
from earth2studio.models.px import GraphCastOperational
from earth2studio.data.rx import LandSeaMask, SurfaceGeoPotential
from earth2studio.run import deterministic
from earth2studio.io import NetCDF4Backend
from earth2studio.io.zarr import ZarrBackend


class MemoryDataSource(DataSource):
    def __init__(self, data: xr.DataArray):
        super().__init__()
        self.data = data
    def __call__(self, init_time, variable, **kwargs):
        return self.data

def main(init_time, model):
    # The initialization time for the forecast
    # Note: GEFS data is available at 00, 06, 12, 18 UTC.
    # Let's use a init_time in the past for reproducibility.
    forecast_length = 240
    forecast_step = 6
    nsteps = forecast_length//forecast_step

    # Define the forecast lead times. We will run a 5-day forecast.
    # Pangu 6-hour model requires lead times in multiples of 6 hours.
    # Define the output directory for the ensemble forecasts
    output_dir = f"/glade/derecho/scratch/ahijevyc/ai-models/output/graphcast/{init_time:%Y%m%d%H}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Ensemble forecast outputs will be saved in: {output_dir}")

    print("Initializing model...")
    model = model.load_model(model.load_default_package())
    # Get the grid the model expects by constructing it from its input coordinates
    lat = model.input_coords()['lat']
    lon = model.input_coords()['lon']

    # align with model input coords
    z = SurfaceGeoPotential(cache=False)([init_time]).sel(lat=lat, lon=lon)
    z["variable"] = ['z']  # rename zsl->z
    lsm = LandSeaMask(cache=False)([init_time]).sel(lat=lat, lon=lon)
    model_variables = model.input_coords()['variable']
    vars_to_zero_fill = [v for v in model_variables if v.startswith('w') or v == 'tp06']

    vars_to_fetch = [v for v in model_variables if v not in vars_to_zero_fill]
    vars_to_fetch.remove('z')
    vars_to_fetch.remove('lsm')

    # 4. The Main Ensemble Loop
    for i, member in enumerate(['gec00'] + [f'gep{p:02d}' for p in range(1,31)]):
        # Define a unique output path for this member's forecast
        output_path = os.path.join(output_dir, member)
        # Use Zarr as the I/O backend takes 20x longer than Netcdf
        #writer = ZarrBackend(file_name=output_path, backend_kwargs={"overwrite": True})
        if os.path.exists(output_path+".nc"):
            os.remove(output_path+".nc")
        writer = NetCDF4Backend(output_path+".nc", backend_kwargs={"mode": "w"})

        # 4a. Fetch initial conditions for the current member
        print(f"Fetching initial conditions for {member} at {init_time.isoformat()}...")
        gefs = GEFS_FX(member=member)
        initial_state_partial = gefs(init_time, [datetime.timedelta(hours=0)], vars_to_fetch)

        # Manually regrid the data to the target grid
        print(f"Regridding initial state for {member}...")
        # Deal with wrapped longitude. Copy 0 deg to 360 deg so points are all valid after interp.
        wrapped = initial_state_partial.sel(lon=0).assign_coords(lon=360)
        initial_state_periodic = xr.concat([initial_state_partial, wrapped], dim="lon")
        # Now interpolate
        initial_state_partial = initial_state_periodic.interp(
            lat=lat, lon=lon, method="linear"
        )

        data_arrays_to_concat = [initial_state_partial]
        for var_name in vars_to_zero_fill:
            zero_array = xr.zeros_like(initial_state_partial.isel(variable=0))
            zero_array['variable'] = var_name
            data_arrays_to_concat.append(zero_array)
        data_arrays_to_concat.extend([z, lsm])


        initial_state = xr.concat(data_arrays_to_concat, dim="variable", coords='minimal')
        assert initial_state.notnull().all()
        # squeeze and drop to avoid ValueError: Dimension lead_time already exists.
        initial_state = initial_state.sel(variable=model_variables).squeeze(dim="lead_time", drop=True)
        # 4b. Run the forecast using the fetched initial state
        in_memory_source = MemoryDataSource(initial_state)
        deterministic([init_time], nsteps, model, in_memory_source, writer)
        writer.close()


        print(f"--- Finished forecast for member '{member}' ---")
    print("\n✅ All ensemble member forecasts have been successfully generated.")

if __name__ == '__main__':
    init_time = pd.to_datetime(sys.argv[1])
    main(init_time, GraphCastOperational)
