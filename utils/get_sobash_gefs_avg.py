# %%
import os
import dataclasses
import numpy as np
import xarray as xr
import concurrent.futures
from typing import List, Union
from tqdm import tqdm
import shutil
import warnings
import dataclasses
import datetime
from dateutil.relativedelta import relativedelta
import pygrib
import multiprocessing as mp
from functools import partial


# Define the pressure levels in reverse order
pressure_levels = ['50', '100', '200', '250', '500', '700', '850', '925', '1000']

# Define the variables that need to be flipped
variables = ['z', 'q', 't', 'u', 'v']

# Create the flipped part of the list
flipped_channels = [f"{var}{level}" for var in variables for level in pressure_levels]

# Define the surface variables
surface_variables = ['msl', 'u10m', 'v10m', 't2m', 'sp', 'mlcape', 'mlcin', 'r2m', 'pwat']

# Combine the flipped channels with the surface variables
HRES_CHANNEL_AVAIL = flipped_channels + surface_variables

## The total precipitation defaults to the every 6-hour accum precipitation in the latest version which has code 596, the accum precip for the entire period is 597
COMPOSITE_SFC_VARS_AVAIL = [ "Pressure reduced to MSL", "10 metre U wind component", "10 metre V wind component", "2 metre temperature",
    'Surface pressure', 'Convective available potential energy', 'Convective inhibition', '2 metre relative humidity', 'Precipitable water',
]

# Define paths for GFS data
GFS_DATA_PATH = "/glade/derecho/scratch/sobash/GEFS/"  #"/glade/campaign/collections/rda/data/ds084.1/"

# Fixed version - Create separate arrays for each variable type
PRESSURE_LEVELS = list(reversed([1000, 925, 850, 700, 500, 250, 200, 100, 50]))

GFS_LEVEL = (
    PRESSURE_LEVELS +  # for Geopotential height (z)
    PRESSURE_LEVELS +  # for Specific humidity (q)
    PRESSURE_LEVELS +  # for Temperature (t)
    PRESSURE_LEVELS +  # for U component of wind (u)
    PRESSURE_LEVELS    # for V component of wind (v)
)

GFS_LEVEL_TYPE = ["isobaricInhPa"] * len(GFS_LEVEL)


# %%
GFS_LEVEL_NAME =["Geopotential height"] * 9 +["Specific humidity"] * 9 + ["Temperature"] * 9 + ["U component of wind"] * 9 + ["V component of wind"] * 9


# %%


assert len(GFS_LEVEL) == len(GFS_LEVEL_TYPE) == len(GFS_LEVEL_NAME)


# %%


GFS_SFC_LEVEL = [
    0, 10, 10, 2,
    0, 18000 , 18000, 2, 0,
    ]
GFS_SFC_LEVEL_TYPE = [
    "meanSea", "heightAboveGround","heightAboveGround","heightAboveGround",
    'surface', 'pressureFromGroundLayer', 'pressureFromGroundLayer', 'heightAboveGround','atmosphereSingleLayer',
]

assert len(GFS_SFC_LEVEL) == len(GFS_SFC_LEVEL_TYPE) == len(COMPOSITE_SFC_VARS_AVAIL)
GFS_LEVELS = GFS_LEVEL + GFS_SFC_LEVEL
GFS_TYPES = GFS_LEVEL_TYPE + GFS_SFC_LEVEL_TYPE
GFS_NAMES = GFS_LEVEL_NAME + COMPOSITE_SFC_VARS_AVAIL

# map GFS_NAMES and GFS_LEVELS to PANGU_CHANNEL
CHANNEL_MAPPING = {channel: (name, level, type_) for channel, name, level, type_ in zip(HRES_CHANNEL_AVAIL, GFS_NAMES, GFS_LEVELS, GFS_TYPES)}

def get_gefsavg_info(channels):
    # Separate pressure level variables and surface variables
    pressure_vars = []
    surface_vars = []
    
    for channel in channels:
        name, level, type_ = CHANNEL_MAPPING[channel]
        if type_ == "isobaricInhPa":
            pressure_vars.append((channel, name, level, type_))
        else:
            surface_vars.append((channel, name, level, type_))
    
    # Sort pressure variables by level (ascending)
    pressure_vars.sort(key=lambda x: x[2])
    
    # Combine sorted pressure variables with surface variables
    sorted_vars = pressure_vars + surface_vars
    
    # Unzip the sorted variables
    if sorted_vars:
        sorted_channels, sorted_names, sorted_levels, sorted_types = zip(*sorted_vars)
    else:
        return [], [], []
    
    return list(sorted_levels), list(sorted_types), list(sorted_names), list(sorted_channels)

def relative_humidity_to_specific_humidity(relative_humidity, temperature, pressure_hPa):
    """
    Convert relative humidity to specific humidity.

    Parameters:
    - relative_humidity: Relative humidity (in percentage)
    - temperature: Temperature in Kelvin
    - pressure: Air pressure in hPa

    Returns:
    - specific_humidity: Specific humidity (in kg/kg)
    """
    # Constants
    E0 = 6.112  # hPa
    a = 17.67
    b = 243.5  # °C
    # Convert Kelvin to Celsius
    temperature = temperature - 273.15

    # Calculate saturation vapor pressure using Magnus formula
    saturation_vapor_pressure = E0 * np.exp((a * temperature) / (b + temperature))

    # Calculate actual vapor pressure
    actual_vapor_pressure = (relative_humidity / 100.0) * saturation_vapor_pressure

    # Calculate specific humidity
    specific_humidity = (0.622 * actual_vapor_pressure) / (pressure_hPa - (0.378 * actual_vapor_pressure))

    return specific_humidity



def process_file(time, channels, forecast_time):
    year = str(time.year)
    month = str(time.month).zfill(2)
    day = str(time.day).zfill(2)
    hour = str(time.hour).zfill(2)
    forecast_time_str = str(forecast_time).zfill(3)
    # file_path = f"{GFS_DATA_PATH}/{year}{month}{day}00/gfs.0p25.{year}{month}{day}{hour}.f{forecast_time_str}.grib2"
    file_path = f"{GFS_DATA_PATH}/{year}{month}{day}00/geavg.t00z.pgrb2a.0p50.f{forecast_time_str}"
    
    if not os.path.exists(file_path):
        raise TypeError("NO DATA TYPE FOUND.")

    gfs_levels, gfs_types, gfs_names, sorted_channels = get_gefsavg_info(channels)
    # print(sorted_channels, gfs_levels, gfs_types, gfs_names)
    results = []
    
    with pygrib.open(file_path) as fh:
        grb = fh.readline()
        init_date = grb.analDate
        valid_date = grb.validDate
        lats, lons = grb.latlons()
    
        # Group fields by name
        field_groups = {}
        for i, (name, level_type, level) in enumerate(zip(gfs_names, gfs_types, gfs_levels)):
            if name not in field_groups:
                field_groups[name] = {"indices": [], "level_type": level_type, "levels": []}
            field_groups[name]["indices"].append(i)
            field_groups[name]["levels"].append(level)
    
        for name, group in field_groups.items():
            if name == "Specific humidity":
                try:
                    fields = fh.select(name=name, typeOfLevel=group["level_type"], level=group["levels"])
                    for i, field in zip(group["indices"], fields):
                        results.append((i, field.values))
                    # print('Specific humidity available')
                except:
                    # print(group['levels'])
                    rh_fields = fh.select(name="Relative humidity", typeOfLevel=group["level_type"], level=group["levels"])
                    t_fields = fh.select(name="Temperature", typeOfLevel=group["level_type"], level=group["levels"])
                    for i, rh_field, t_field, level in zip(group["indices"], rh_fields, t_fields, group["levels"]):
                        # print(rh_field)
                        field = relative_humidity_to_specific_humidity(rh_field.values, t_field.values, level)
                        results.append((i, field))
                    # print('Specific humidity not available, convert from RH')
            elif name == "Geopotential height":
                fields = fh.select(name=name, typeOfLevel=group["level_type"], level=group["levels"])
                for i, field in zip(group["indices"], fields):
                    field_values = field.values * 9.80665
                    results.append((i, field_values))
            else:
                fields = fh.select(name=name, typeOfLevel=group["level_type"], level=group["levels"])    
                # print(fields)
                for i, field in zip(group["indices"], fields):
                    results.append((i, field.values))

    return results, sorted_channels, init_date, valid_date, lats[:, 0], lons[0, :]

def open_gefsavg_nc(time, channels, forecast_time):

    results, sorted_channels, init_date, valid_date, lats, lons = process_file(time, channels, forecast_time)

    data = np.empty((len(sorted_channels), 361, 720))
    for i, field in results:
        data[i] = field

    dataarray_ls = xr.DataArray(data, dims=["channel", "lat", "lon"])
    dataarray_ls = dataarray_ls.assign_coords(time=time, channel=sorted_channels, lat=lats, lon=lons).expand_dims("time") #.transpose("time", "channel", "lat", "lon")
    return dataarray_ls

def _get_channels(time: datetime, channels: List[str], forecast_time: int):
    if not isinstance(channels, list):
        raise TypeError("channels must be a list")

    darray = open_gefsavg_nc(time, channels, forecast_time)
    return darray

@dataclasses.dataclass
class GEFSAVGDataSource:
    channel_names: List[str]
    forecast_time: int

    @property
    def time_means(self):
        raise NotImplementedError()

    def __call__(self, time: datetime):
        return _get_channels(time, self.channel_names, self.forecast_time)



if __name__ == "__main__":
    pangu_channel = [
        'z1000', 'z925', 'z850', 'z700', 'z500', 'z250', 'z200', 'z100', 'z50', 'q1000',
        'q925', 'q850', 'q700', 'q500', 'q250', 'q200', 'q100', 'q50', 't1000', 't925',
        't850', 't700', 't500', 't250', 't200', 't100', 't50', 'u1000', 'u925', 'u850',
        'u700', 'u500', 'u250', 'u200', 'u100', 'u50', 'v1000', 'v925', 'v850', 'v700',
        'v500', 'v250', 'v200', 'v100', 'v50']
    # pangu_channel = ['t925','t1000','t500','q500','msl', 'u10m', 'v10m', 't2m','sp','mlcape','mlcin','pwat','r2m']
    forecast_time = 24  # Example forecast time in hours
    ds = GEFSAVGDataSource(pangu_channel, forecast_time)
    res = ds(datetime.datetime(2021, 1, 1, 0))
    # print(res)
    print(res.isel(time=0).sel(channel='t1000').values)
    
    print(res.isel(time=0).sel(channel='t925').values)
    
    print(np.nanmean(res.isel(time=0).sel(channel='t1000').values-res.isel(time=0).sel(channel='t925').values))
    # print(res.sel(channel=['q500']).values)
