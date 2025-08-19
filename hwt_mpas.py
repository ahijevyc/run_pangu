# =============================================================================
# Imports
# =============================================================================
import dataclasses
import re
import warnings
from functools import partial
from pathlib import Path
from typing import List, Tuple, Union, Optional

import dask
from dask.distributed import Client
import numpy as np
import pandas as pd
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

# Define the base path for campaign data.
camp_schwartz = Path("/glade/campaign/mmm/parc/schwartz")


# =============================================================================
# Helper Function for Preprocessing
# =============================================================================
def _preprocess_mpas_file(ds: xr.Dataset, valid_time: pd.Timestamp, variables: Optional[List[str]] = None) -> xr.Dataset:
    """
    This standalone function processes a single dataset from one file.
    It's called by open_mfdataset for each file before combining them.
    Making it a top-level function ensures it can be serialized by Dask.
    """
    # If a list of variables is provided, filter the channel mapping.
    # Otherwise, use all available channels.
    if variables:
        local_channel_to_name = {k: v for k, v in CHANNEL_TO_NAME.items() if k in variables}
    else:
        local_channel_to_name = CHANNEL_TO_NAME
    
    # Extract member number from the file path, which xarray stores in ds.encoding
    match = re.search(r"mem_(\d+)", ds.encoding["source"])
    mem = int(match.group(1)) if match else -1

    # --- Perform calculations ---
    ds = xtime(ds)
    # Calculate geopotential from height
    height_vars = [var for var in ds.data_vars if var.startswith("height_")]
    for var in height_vars:
        pl_match = re.search(r"(\d+\.?\d*)hPa$", var)
        if pl_match:
            pl = pl_match.group(1)
            ds[f"geopotential_{pl}hPa"] = ds[var] * g.m
    
    # Calculate mixing ratio from relative humidity
    rh_vars = [var for var in ds.data_vars if var.startswith("relhum_")]
    for var in rh_vars:
        pl_match = re.search(r"(\d+\.?\d*)hPa$", var)
        if pl_match:
            pl = pl_match.group(1)
            pressure = float(pl) * units.hPa
            temperature = ds[f"temperature_{pl}hPa"]
            relative_humidity = ds[var]
            # Dequantify to remove units, which can cause issues with xarray/dask
            mixing_ratio = mixing_ratio_from_relative_humidity(
                pressure, temperature, relative_humidity
            )
            ds[f"mixing_ratio_{pl}hPa"] = mixing_ratio.metpy.dequantify()

    # --- Standardize variable names and dimensions based on the filtered list ---
    das = [ds[name].rename(channel) for channel, name in local_channel_to_name.items() if name in ds]
    ds = xr.merge(das, combine_attrs="drop")

    # Add member and time dimensions for proper concatenation
    ds = ds.expand_dims({"member": [mem], "time": [valid_time]})
    return ds


# =============================================================================
# Main Data Source Class
# =============================================================================
@dataclasses.dataclass
class MPASDataSource:
    """
    A self-contained data source class to fetch, process, and regrid MPAS data.
    This class encapsulates all the logic for a given data source configuration.
    """

    grid_path: Path
    data_dir: Path
    d_lon: float = 0.25
    d_lat: float = 0.25
    grid_ncells: int = dataclasses.field(init=False, repr=False)

    def __post_init__(self):
        """
        Post-initialization hook to set up paths, the target grid,
        and regridding indices.
        """
        # --- Construct full paths ---
        self.grid_path = camp_schwartz / self.grid_path
        self.data_dir = camp_schwartz / self.data_dir

        # --- Define and store the target grid within the class instance ---
        self.new_lon = np.arange(0, 360, self.d_lon)
        self.new_lat = np.arange(-90, 90 + self.d_lat, self.d_lat)[::-1]  # north to south

        # --- Compute or load regridding indices ---
        self.distance, self.indices = self._nearest_indices()

    def _lon_lat_to_cartesian(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Converts longitude and latitude (in radians) to 3D Cartesian coordinates."""
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)
        return np.array([x, y, z]).T

    def _nearest_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates or loads cached nearest neighbor indices for regridding."""
        cached = (Path("/glade/derecho/scratch/ahijevyc/ai-models") / self.grid_path.name).with_suffix(".npz")
        if cached.exists():
            print(f"Loading cached indices from {cached}")
            loaded_data = np.load(cached)
            grid = xr.open_dataset(self.grid_path)
            self.grid_ncells = grid.sizes["nCells"]
            return loaded_data["dists"], loaded_data["inds"]

        new_lon_grid, new_lat_grid = np.meshgrid(self.new_lon, self.new_lat)
        print("Building KDTree from MPAS grid cells...")
        grid = xr.open_dataset(self.grid_path)
        self.grid_ncells = grid.sizes["nCells"] 
        mpas_xyz = self._lon_lat_to_cartesian(grid["lonCell"].values, grid["latCell"].values)

        target_lon_rad = np.deg2rad(new_lon_grid.ravel())
        target_lat_rad = np.deg2rad(new_lat_grid.ravel())
        target_xyz = self._lon_lat_to_cartesian(target_lon_rad, target_lat_rad)

        kdtree = KDTree(mpas_xyz)
        print("Querying tree to find nearest neighbors for regridding map...")
        distance, indices = kdtree.query(target_xyz)
        print(f"Saving indices to {cached}")
        np.savez_compressed(cached, dists=distance, inds=indices)
        return distance, indices

    def _open_casper_nc(self, itime: pd.Timestamp, fhr: int, variables: Optional[List[str]] = None) -> xr.Dataset:
        """
        Opens and concatenates data from multiple ensemble members in parallel
        using xr.open_mfdataset for efficiency.
        """
        valid_time = itime + pd.Timedelta(hours=fhr)
        data_paths = [
            self.data_dir / f"{itime.strftime('%Y%m%d%H')}/post/mem_{mem}/diag.{valid_time.strftime('%Y-%m-%d_%H.%M.%S')}.nc"
            for mem in range(1, 11)
        ]
        
        existing_paths = [p for p in data_paths if p.exists()]
        if not existing_paths:
            raise FileNotFoundError(f"Could not find any member files for itime={itime}, fhr={fhr}.")

        preprocessor = partial(_preprocess_mpas_file, valid_time=valid_time, variables=variables)

        return xr.open_mfdataset(
            existing_paths,
            combine="nested",
            concat_dim="member",
            preprocess=preprocessor,
            parallel=True,
            engine="netcdf4"
        )

    def _regrid_dataset(self, ds_mpas: xr.Dataset) -> xr.Dataset:
        """
        Remaps the dataset from the unstructured grid to a regular lat-lon grid
        using a vectorized approach.
        """
        da_mpas = ds_mpas.to_dataarray(dim="channel")
        #print("Loading data into memory before regridding...")
        da_mpas = da_mpas.load()
        #print("Data loaded.")

        remapped_flat = da_mpas.isel(nCells=self.indices)
        other_dims = {k: v for k, v in da_mpas.sizes.items() if k != "nCells"}
        new_shape = tuple(other_dims.values()) + (len(self.new_lat), len(self.new_lon))
        reshaped_data = remapped_flat.values.reshape(new_shape)

        final_dims = tuple(other_dims.keys()) + ("lat", "lon")
        final_coords = {dim: da_mpas[dim] for dim in other_dims.keys()}
        final_coords["lat"] = self.new_lat
        final_coords["lon"] = self.new_lon

        da_remapped = xr.DataArray(
            data=reshaped_data,
            coords=final_coords,
            dims=final_dims,
            attrs=da_mpas.attrs,
        )

        ds_remapped = da_remapped.to_dataset(dim="channel")
        ds_remapped.attrs = ds_mpas.attrs
        return ds_remapped

    def __call__(self, itime: pd.Timestamp, fhr: int, variables: Optional[List[str]] = None) -> xr.Dataset:
        """
        Makes the class instance callable. This is the main entry point for the workflow.
        """
        
        ds_mpas = self._open_casper_nc(itime, fhr, variables=variables)

        if "nCells" in ds_mpas.dims and ds_mpas.sizes["nCells"] != self.grid_ncells:
            raise ValueError(
                f"Grid mismatch: The grid file '{self.grid_path.name}' has {self.grid_ncells} cells, "
                f"but the data files have {ds_mpas.sizes['nCells']} cells. "
                "The regridding indices are invalid for this data."
            )

        return self._regrid_dataset(ds_mpas)

