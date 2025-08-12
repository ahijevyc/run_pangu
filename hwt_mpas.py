# =============================================================================
# Imports
# =============================================================================
import concurrent.futures
import dataclasses
import re
import warnings
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import uxarray
import xarray as xr
from metpy.calc import mixing_ratio_from_relative_humidity
from metpy.constants import g
from metpy.units import units
from run_pangu.utils.xtime import xtime
from scipy.spatial import KDTree

# =============================================================================
# Configuration
# =============================================================================
# Define the pressure levels for which to extract data.
pressure_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

# Map standard channel names to the variable names in the source NetCDF files.
CHANNEL_TO_NAME = {"t2m": "t2m", "u10m": "u10", "v10m": "v10", "msl": "surface_pressure"}

# Dynamically populate the channel mapping for different pressure levels.
# This section remaps certain pressure levels to their nearest neighbors
# to align with the available data.
for pl in pressure_levels:
    pl_new = pl
    if pl < 200:
        pl_new = 200
    if pl == 300:
        pl_new = 250
    if pl == 400:
        pl_new = 500
    if pl == 600:
        pl_new = 700
    if pl == 1000:
        pl_new = 925
    CHANNEL_TO_NAME[f"t{pl}"] = f"temperature_{pl_new}hPa"
    CHANNEL_TO_NAME[f"u{pl}"] = f"uzonal_{pl_new}hPa"
    CHANNEL_TO_NAME[f"v{pl}"] = f"umeridional_{pl_new}hPa"
    CHANNEL_TO_NAME[f"z{pl}"] = f"geopotential_{pl_new}hPa"
    CHANNEL_TO_NAME[f"q{pl}"] = f"mixing_ratio_{pl_new}hPa"


# =============================================================================
# Data Loading and Processing Functions
# =============================================================================
def process_name(itime: pd.Timestamp, fhr: int, mem: int) -> Union[xr.Dataset, None]:
    """
    Opens and processes a single ensemble member NetCDF file.

    This function constructs the file path, opens the dataset, selects the
    correct time slice, extracts and renames variables, and merges them.

    Args:
        itime: The initialization time of the forecast.
        fhr: The forecast hour.
        mem: The ensemble member number.

    Returns:
        An xarray.Dataset for the single member, or None if an error occurs.
    """
    data_path = f"/glade/campaign/mmm/parc/schwartz/HWT{itime.year}/mpas_15km"
    valid_time = itime + pd.Timedelta(hours=fhr)
    path = f"{data_path}/{itime.strftime('%Y%m%d%H')}/post/mem_{mem}/diag.{valid_time.strftime('%Y-%m-%d_%H.%M.%S')}.nc"

    try:
        ds = xr.open_dataset(path)
        ds = xtime(ds)
        # Multiply height by gravity to get geopotential
        height_vars = [var for var in ds.data_vars if var.startswith("height_")]
        for var in height_vars:
            pl = re.search(r"(\d+\.?\d*)hPa$", var).group(1)
            ds[f"geopotential_{pl}hPa"] = ds[var] * g
        # Derive mixing ratio from relative humidity
        rh_vars = [var for var in ds.data_vars if var.startswith("relhum_")]
        for var in rh_vars:
            pl = re.search(r"(\d+\.?\d*)hPa$", var).group(1)
            pressure = float(pl) * units.hPa
            temperature = ds[f"temperature_{pl}hPa"]
            relative_humidity = ds[var]
            ds[f"mixing_ratio_{pl}hPa"] = mixing_ratio_from_relative_humidity(
                pressure, temperature, relative_humidity
            )
        ds = ds.expand_dims({"member": [mem], "time": [valid_time]})

        # Extract and rename all specified channels
        das = [ds[name].rename(channel) for channel, name in CHANNEL_TO_NAME.items()]
        ds = xr.merge(das, combine_attrs="drop")
        return ds
    except FileNotFoundError:
        warnings.warn(f"File not found: {path}")
        return None
    except Exception as e:
        warnings.warn(f"Error processing file {path}: {e}")
        return None


def open_casper_nc(itime: pd.Timestamp, fhr: int) -> xr.Dataset:
    """
    Opens and concatenates data from multiple ensemble members in parallel.

    Args:
        itime: The initialization time of the forecast.
        fhr: The forecast hour.

    Returns:
        A single xarray.Dataset with data from all members concatenated
        along the 'member' dimension.
    """
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        # Submit all file processing tasks to the thread pool
        futures = [executor.submit(process_name, itime, fhr, mem) for mem in range(1, 11)]

    # Collect results and filter out any that failed (returned None)
    results = [future.result() for future in futures]
    valid_results = [res for res in results if res is not None]

    if not valid_results:
        raise FileNotFoundError("Could not open any member files.")

    # Combine the list of datasets into a single dataset
    return xr.concat(valid_results, dim="member")


# =============================================================================
# Regridding Setup and Functions
# =============================================================================
def lon_lat_to_cartesian(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Converts longitude and latitude (in radians) to 3D Cartesian coordinates."""
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.array([x, y, z]).T


# --- Define the Target Latitude-Longitude Grid ---
d_lon, d_lat = 0.25, 0.25
new_lon = np.arange(0, 360, d_lon)
new_lat = np.arange(-90, 90 + d_lat, d_lat)
grid_path = Path("/glade/campaign/mmm/parc/schwartz/MPAS/15km_mesh/grid_mesh/x1.2621442.grid.nc")
cached = (Path("/glade/derecho/scratch/ahijevyc/ai-models") / grid_path.name).with_suffix(".npz")

if cached.exists():
    loaded_data = np.load(cached)
    distance = loaded_data["dists"]
    indices = loaded_data["inds"]
else:
    new_lon_grid, new_lat_grid = np.meshgrid(new_lon, new_lat)

    # --- Build KDTree from the Unstructured MPAS Grid for Fast Regridding ---
    print("Building KDTree from MPAS grid cells...")
    grid = xr.open_dataset(grid_path)
    mpas_xyz = lon_lat_to_cartesian(grid["lonCell"].values, grid["latCell"].values)

    # Convert target grid to Cartesian coordinates for the query
    target_lon_rad = np.deg2rad(new_lon_grid.ravel())
    target_lat_rad = np.deg2rad(new_lat_grid.ravel())
    target_xyz = lon_lat_to_cartesian(target_lon_rad, target_lat_rad)

    # Create the tree and find the nearest MPAS cell for each target grid point
    kdtree = KDTree(mpas_xyz)
    print("Querying tree to find nearest neighbors for regridding map...")
    distance, indices = kdtree.query(target_xyz)
    np.savez_compressed(cached, dists=distance, inds=indices)


def _get_channels(itime: pd.Timestamp, fhr: int) -> xr.Dataset:
    """
    Retrieves, processes, and regrids the MPAS data.

    This function orchestrates the data pipeline: loading the raw MPAS data,
    remapping it from the unstructured grid to a regular lat-lon grid using
    the pre-computed KDTree indices, and returning the final dataset.
    """
    ds_mpas = open_casper_nc(itime, fhr)

    print("Start regridding...")
    remapped_vars = []
    for var_name in ds_mpas.data_vars:
        mpas_da = ds_mpas[var_name]

        # Keep all dims but 'nCells'
        other_dims = [dim for dim in mpas_da.dims if dim != "nCells"]

        # Use the pre-computed indices to select data corresponding to the new grid
        remapped_flat = mpas_da.isel(nCells=indices)

        # Reshape the flattened data back to the 2D grid dimensions
        other_dim_sizes = [mpas_da.sizes[dim] for dim in other_dims]
        n_lat, n_lon = len(new_lat), len(new_lon)
        new_shape = tuple(other_dim_sizes + [n_lat, n_lon])
        reshaped_data = remapped_flat.values.reshape(new_shape)

        # Create the final, remapped DataArray with correct coordinates
        final_coords = {dim: ds_mpas[dim] for dim in other_dims}
        final_coords["lat"] = new_lat
        final_coords["lon"] = new_lon
        final_dims = other_dims + ["lat", "lon"]

        da_remapped = xr.DataArray(
            data=reshaped_data,
            coords=final_coords,
            dims=final_dims,
            name=var_name,
            attrs=mpas_da.attrs,
        )
        remapped_vars.append(da_remapped)

    # Combine the list of remapped DataArrays into a single Dataset
    ds_remapped = xr.merge(remapped_vars, combine_attrs="drop").to_dataarray("channel")
    ds_remapped.attrs = ds_mpas.attrs

    return ds_remapped


# =============================================================================
# Main Execution Class and Block
# =============================================================================
@dataclasses.dataclass
class MPASDataSource:
    """A data source class to fetch and process MPAS data."""

    fhr: int

    def __call__(self, itime: pd.Timestamp) -> xr.Dataset:
        """Makes the class instance callable."""
        return _get_channels(itime, self.fhr)
