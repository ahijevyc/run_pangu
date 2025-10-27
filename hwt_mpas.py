# =============================================================================
# Imports
# =============================================================================
import dataclasses
import logging
import re
from functools import partial
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
from metpy.calc import mixing_ratio_from_relative_humidity
from metpy.constants import g
from metpy.units import units
from utils.xtime import xtime
from scipy.spatial import KDTree

from earth2studio.data import DataSource
from earth2studio.io import IOBackend
from earth2studio.utils.type import CoordSystem

# =============================================================================
# Logging Configuration
# =============================================================================
# Use the logging module for better control over output
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# =============================================================================
# Configuration
# =============================================================================
# Define the pressure levels for which to extract data.
PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]


def get_channel_map(pressure_levels: List[int]) -> dict:
    """
    Generates the mapping from standard channel names to source NetCDF variable names.

    The pressure level remapping (e.g., 600 -> 700) is done to align the MPAS
    output with the specific input requirements of a downstream model like Pangu-Weather,
    which may be trained on a different set of standard pressure levels.
    """
    channel_map = {
        "t2m": "t2m",
        "u10m": "u10",
        "v10m": "v10",
        "msl": "surface_pressure"
    }
    for pl in pressure_levels:
        # Determine the target pressure level for variable name construction
        if pl <= 200:
            target_pl = 200
        elif pl == 300:
            target_pl = 250
        elif pl == 400:
            target_pl = 500
        elif pl == 600:
            target_pl = 700
        elif pl == 1000:
            target_pl = 925
        else:
            target_pl = pl

        channel_map[f"t{pl}"] = f"temperature_{target_pl}hPa"
        channel_map[f"u{pl}"] = f"uzonal_{target_pl}hPa"
        channel_map[f"v{pl}"] = f"umeridional_{target_pl}hPa"
        channel_map[f"z{pl}"] = f"geopotential_{target_pl}hPa"
        channel_map[f"q{pl}"] = f"mixing_ratio_{target_pl}hPa"
    return channel_map


# Generate the channel mapping
CHANNEL_TO_NAME = get_channel_map(PRESSURE_LEVELS)


# =============================================================================
# Helper Function for Preprocessing
# =============================================================================
def _preprocess_mpas_file(
    ds: xr.Dataset, valid_time: pd.Timestamp, variables: Optional[List[str]] = None
) -> xr.Dataset:
    """
    This standalone function processes a single dataset from one file.
    It's called by open_mfdataset for each file before combining them.
    Making it a top-level function ensures it can be serialized by Dask.
    """

    # Extract member number from the file path
    match = re.search(r"mem_(\d+)", ds.encoding["source"])
    mem = int(match.group(1)) if match else -1

    # --- Perform calculations ---
    ds = xtime(ds)

    # Calculate geopotential from height, if not already present
    height_vars = [var for var in ds.data_vars if var.startswith("height_")]
    for var in height_vars:
        pl_match = re.search(r"(\d+\.?\d*)hPa$", var)
        if pl_match:
            pl = pl_match.group(1)
            geopotential_var = f"geopotential_{pl}hPa"
            if geopotential_var not in ds:
                ds[geopotential_var] = ds[var] * g.m

    # Calculate mixing ratio from relative humidity, if not already present
    rh_vars = [var for var in ds.data_vars if var.startswith("relhum_")]
    for var in rh_vars:
        pl_match = re.search(r"(\d+\.?\d*)hPa$", var)
        if pl_match:
            pl = pl_match.group(1)
            mixing_ratio_var = f"mixing_ratio_{pl}hPa"
            if mixing_ratio_var not in ds:
                pressure = float(pl) * units.hPa
                temperature = ds[f"temperature_{pl}hPa"]
                relative_humidity = ds[var]
                mixing_ratio = mixing_ratio_from_relative_humidity(
                    pressure, temperature, relative_humidity
                )
                ds[mixing_ratio_var] = mixing_ratio.metpy.dequantify()

    # If `variables` is None, default to processing all channels defined in the map.
    channels_to_process = variables if variables is not None else list(CHANNEL_TO_NAME.keys())
    
    # Create a list of xarray DataArrays to be merged.
    das = []
    # Loop through the list of channels to process.
    for channel in channels_to_process:
        source_name = CHANNEL_TO_NAME.get(channel)
        if source_name is not None and source_name in ds:
            # If the variable exists, rename it and add to the list.
            das.append(ds[source_name].rename(channel))
        else:
            # If the variable does not exist, create a zero-filled DataArray
            # with the correct dimensions and add it to the list.
            logging.warning(f"Variable '{channel}' not found. Returning a zero array.")
            # Determine the appropriate size for the zero array.
            # Assuming 'nCells' is the only dimension that varies.
            if "nCells" in ds.dims:
                zero_array = np.zeros(ds.sizes["nCells"])
                dims = ("nCells",)
            else:
                # Fallback for other dimensions if needed.
                zero_array = np.zeros(())
                dims = ()
            da_zero = xr.DataArray(zero_array, dims=dims, name=channel)
            das.append(da_zero)
            
    # Merge all the DataArrays into a single Dataset.
    ds = xr.merge(das, combine_attrs="drop")

    # Add member and time dimensions for proper concatenation
    ds = ds.expand_dims({"member": [mem], "time": [valid_time]})
    return ds


# =============================================================================
# Main Data Source Class
# =============================================================================
@dataclasses.dataclass
class MPASEnsDataSource:
    """
    A self-contained data source class to fetch, process, and regrid MPAS data.
    Returns multiple members of the ensemble (all 10 by default).
    Might want to use earth2studio.data.MPASDataSource instead. It doesn't do
    multiple members but it works well in the earth2studio framework.

    Attributes:
        grid_path: Relative path to the MPAS grid definition file.
        data_dir: Relative path to the directory containing MPAS output.
        members: A list of ensemble member numbers to process.
        base_path: The base directory for campaign data.
        cache_path: Directory to store cached regridding indices.
        d_lon: Target grid longitude spacing.
        d_lat: Target grid latitude spacing.
    """

    grid_path: Path
    data_dir: Path
    members: List[int] = dataclasses.field(default_factory=lambda: list(range(1, 11)))
    base_path: Path = Path("/glade/campaign/mmm/parc/schwartz")
    cache_path: Path = Path("/glade/derecho/scratch/ahijevyc/ai-models")
    d_lon: float = 0.25
    d_lat: float = 0.25
    grid_ncells: int = dataclasses.field(init=False, repr=False)

    def __post_init__(self):
        """
        Post-initialization hook to set up paths, the target grid,
        and regridding indices.
        """
        # --- Construct full paths ---
        self.grid_path = self.base_path / self.grid_path
        self.data_dir = self.base_path / self.data_dir
        self.cache_path.mkdir(parents=True, exist_ok=True)

        # --- Define and store the target grid ---
        self.new_lon = np.arange(0, 360, self.d_lon)
        self.new_lat = np.arange(90, -90 - self.d_lat, -self.d_lat)

        # --- Compute or load regridding indices ---
        self.distance, self.indices = self._nearest_indices()

    def _lon_lat_to_cartesian(
        self, lon: np.ndarray, lat: np.ndarray
    ) -> np.ndarray:
        """Converts longitude and latitude (in radians) to 3D Cartesian coords."""
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)
        return np.array([x, y, z]).T

    def _nearest_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates or loads cached nearest neighbor indices for regridding."""
        cached_file = (self.cache_path / self.grid_path.name).with_suffix(".npz")
        if cached_file.exists():
            logging.info(f"Loading cached indices from {cached_file}")
            loaded_data = np.load(cached_file)
            with xr.open_dataset(self.grid_path) as grid:
                self.grid_ncells = grid.sizes["nCells"]
            return loaded_data["dists"], loaded_data["inds"]

        new_lon_grid, new_lat_grid = np.meshgrid(self.new_lon, self.new_lat)
        logging.info("Building KDTree from MPAS grid cells...")
        with xr.open_dataset(self.grid_path) as grid:
            self.grid_ncells = grid.sizes["nCells"]
            mpas_xyz = self._lon_lat_to_cartesian(
                grid["lonCell"].values, grid["latCell"].values
            )

        target_lon_rad = np.deg2rad(new_lon_grid.ravel())
        target_lat_rad = np.deg2rad(new_lat_grid.ravel())
        target_xyz = self._lon_lat_to_cartesian(target_lon_rad, target_lat_rad)

        kdtree = KDTree(mpas_xyz)
        logging.info("Querying tree to find nearest neighbors for regridding...")
        distance, indices = kdtree.query(target_xyz)

        logging.info(f"Saving indices to {cached_file}")
        np.savez_compressed(cached_file, dists=distance, inds=indices)
        return distance, indices

    def _open_casper_nc(
        self, itime: pd.Timestamp, fhr: int, members: List[int], variables: Optional[List[str]] = None
    ) -> xr.Dataset:
        """
        Opens and concatenates data from multiple ensemble members in parallel.
        """
        valid_time = itime + pd.Timedelta(hours=fhr)
        data_paths = [
            self.data_dir
            / f"{itime.strftime('%Y%m%d%H')}/post/mem_{mem}/diag.{valid_time.strftime('%Y-%m-%d_%H.%M.%S')}.nc"
            for mem in members
        ]

        existing_paths = [p for p in data_paths if p.exists()]
        if not existing_paths:
            raise FileNotFoundError(
                f"Could not find any member files for itime={itime}, fhr={fhr} for members {members}."
            )
        logging.info(f"Found {len(existing_paths)} member files to open.")

        preprocessor = partial(
            _preprocess_mpas_file, valid_time=valid_time, variables=variables
        )

        return xr.open_mfdataset(
            existing_paths,
            combine="nested",
            concat_dim="member",
            preprocess=preprocessor,
            parallel=True,
            engine="netcdf4",
        )

    def _regrid_dataset(self, ds_mpas: xr.Dataset) -> xr.Dataset:
        """
        Remaps the dataset from the unstructured grid to a regular lat-lon grid.
        """
        da_mpas = ds_mpas.to_dataarray(dim="channel")

        # This is a memory-intensive step. It loads the full dataset into RAM.
        logging.info("Loading data into memory before regridding...")
        da_mpas = da_mpas.load()
        logging.info("Data loaded. Performing regridding.")

        regridded_data = da_mpas.isel(nCells=self.indices)
        
        # This part of the code needs to handle a potential missing 'nCells' dimension
        # when a zero-array is created without it (e.g., if a 2D variable is requested)
        # We need to reshape the output to the target grid.
        dims = list(regridded_data.dims)
        try:
            ncells_index = dims.index('nCells')
            dims.pop(ncells_index)
            dims.extend(['lat', 'lon'])
            new_shape = tuple(regridded_data.sizes[d] for d in dims[:-2]) + (
                len(self.new_lat),
                len(self.new_lon),
            )
            reshaped_data = regridded_data.values.reshape(new_shape)
        except ValueError:
            # If 'nCells' is not a dimension, the data is already a scalar or 
            # has other dimensions. Reshaping to a grid of zeros.
            logging.warning("Reshaping a non-grid variable to a full grid of zeros.")
            channel_size = regridded_data.sizes["channel"]
            new_shape = (channel_size, len(self.new_lat), len(self.new_lon))
            reshaped_data = np.zeros(new_shape)
            dims = ['channel', 'lat', 'lon']


        final_coords = {dim: regridded_data.coords[dim] for dim in dims[:-2]}
        final_coords["lat"] = self.new_lat
        final_coords["lon"] = self.new_lon

        da_remapped = xr.DataArray(
            data=reshaped_data,
            coords=final_coords,
            dims=dims,
            attrs=regridded_data.attrs,
        )

        ds_remapped = da_remapped.to_dataset(dim="channel")
        ds_remapped.attrs = ds_mpas.attrs
        return ds_remapped

    def __call__(
        self, itime: pd.Timestamp, fhr: int, variables: Optional[List[str]] = None, members: Optional[List[int]] = None
    ) -> xr.Dataset:
        """
        Makes the class instance callable. This is the main entry point for the workflow.
        """
        # Use the members provided in the call, otherwise default to the instance's members
        members_to_process = members if members is not None else self.members

        ds_mpas = self._open_casper_nc(itime, fhr, members=members_to_process, variables=variables)
        # Check for grid mismatch only if a variable with nCells exists.
        has_ncells_var = any("nCells" in var.dims for var in ds_mpas.data_vars.values())
        if has_ncells_var and ds_mpas.sizes.get("nCells") != self.grid_ncells:
            raise ValueError(
                f"Grid mismatch: The grid file '{self.grid_path.name}' has {self.grid_ncells} cells, "
                f"but the data files have {ds_mpas.sizes['nCells']} cells. "
                "The regridding indices are invalid for this data."
            )

        ds_regridded = self._regrid_dataset(ds_mpas)

        # Add metadata as global attributes.
        ds_regridded.attrs["initialization_time"] = str(itime)
        ds_regridded.attrs["forecast_hour"] = fhr

        logging.info("Finished processing and regridding.")
        return ds_regridded.to_dataarray(dim="variable")  # GraphCast model, e2s expect 'variable'


class MemoryDataSource(DataSource):
    """A simple data source that holds a single xarray.DataArray state in memory."""

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
