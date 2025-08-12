import os
import dataclasses
import numpy as np
import xarray as xr
from typing import List, Union
from tqdm import tqdm
import dataclasses
import datetime
import pandas as pd

# Define paths for GFS data
wxformer_6h_path = '/glade/campaign/cisl/aiml/ksha/CREDIT_arXiv/GATHER/wxformer_6h_20241029/'
fuxi_6h_path = '/glade/campaign/cisl/aiml/ksha/CREDIT_arXiv/GATHER/fuxi_6h_20241029/'

def sort_and_attach_path(filenames, parent_path):
    # Sort the filenames by date
    sorted_filenames = sorted(filenames, key=lambda x: x[:10])
    
    # Attach the parent path to each sorted filename
    full_paths = [parent_path + filename for filename in sorted_filenames]
    
    return full_paths

fuxi_6h_fpath = sort_and_attach_path(os.listdir(fuxi_6h_path), fuxi_6h_path)
wxformer_6h_fpath = sort_and_attach_path(os.listdir(wxformer_6h_path), wxformer_6h_path)

def preprocess_dataset(ds, init_time, forecast_time):
    # print(init_time)
    # Convert time to prediction_hourdelta
    forecast_hours = ds['forecast_hour'].values
    prediction_timedelta = pd.to_timedelta(forecast_hours, unit='h')
    # print(len(prediction_timedelta))
    ds = ds.assign_coords(prediction_timedelta=('time',  prediction_timedelta))
    
    ds = ds.swap_dims({'time': 'prediction_timedelta'})    
    
    # Add init_time coordinate
    ds = ds.assign_coords(time=[init_time]).drop_vars({'forecast_hour'})

    # Rename variables
    variable_rename = {
        'Q': 'specific_humidity',
        'V': 'v_component_of_wind',
        'U': 'u_component_of_wind',
        'T': 'temperature',
        'SP': 'surface_pressure',
        't2m': '2m_temperature',
        'U500': 'u500',
        'V500': 'v500',
        'T500': 't500',
        'Z500': 'z500',
        'Q500': 'q500'
    }
    ds = ds.rename(variable_rename)
    ds = ds.assign_coords(level=[ 10,  30,  40,  50,  60,  70,  80,  90,  95, 100, 105, 110, 120, 130, 136, 137]).isel(level=slice(None, None, -1))
    ds.coords['longitude'] = (ds.coords['longitude'] + 180) % 360 - 180
    ds = ds.sortby(ds.longitude).isel(latitude=slice(None, None, -1))
    return ds.sel(prediction_timedelta=np.timedelta64(forecast_time, 'h'))

# Example files: '/glade/campaign/cisl/aiml/ksha/CREDIT_arXiv/GATHER/fuxi_6h/2019-01-01T00Z.nc', '/glade/campaign/cisl/aiml/ksha/CREDIT_arXiv/GATHER/fuxi_6h/2019-01-01T12Z.nc'

def process_file(time, forecast_time, model):
    # 1. find the file based on the time
    if model == 'wxformer':
        fpath = wxformer_6h_fpath
    elif model == 'fuxi':
        fpath = fuxi_6h_fpath
    else:
        raise ValueError("model must be 'wxformer' or 'fuxi'")
    # Convert the datetime to the string format used in filenames
    time_str = time.strftime('%Y-%m-%dT%HZ.nc')
    
    # Find the file in the sorted file paths
    file_path = next((fp for fp in fpath if time_str in fp), None)
    
    if file_path is None:
        raise TypeError(f"Data not found for date {time_str}")
    
    # Open the file and preprocess the data similar to weatherbench2 format
    with xr.open_dataset(file_path, chunks={'time': -1, 'level': -1, 'latitude': 'auto', 'longitude': 'auto'}) as ds:
        preprocess_ds = preprocess_dataset(ds, time, forecast_time)
    return preprocess_ds

@dataclasses.dataclass
class AiNWP6hDataSource:
    start_lat: float
    end_lat: float
    start_lon: float
    end_lon: float
    model: str = 'wxformer' # 'wxformer' or 'fuxi'
    
    @property
    def time_means(self):
        raise NotImplementedError()

    def __call__(self, time: datetime, forecast_time: int):
        ds = self._get_ds(time, forecast_time)
        lat_slice = slice(self.start_lat, self.end_lat)
        lon_slice = slice(self.start_lon, self.end_lon)
        result = ds.sel(
            latitude=lat_slice,
            longitude=lon_slice,
        )
        return result

    def _get_ds(self, time: datetime, forecast_time: int):
        if not isinstance(self.model, str):
            raise TypeError("model must be a string")

        ds = process_file(time, forecast_time, self.model)
        
        return ds     
    
if __name__ == "__main__":
    wxformer_channel = ['qL137',
 'qL136',
 'qL130',
 'qL120',
 'qL110',
 'qL105',
 'qL100',
 'qL95',
 'qL90',
 'qL80',
 'qL70',
 'qL60',
 'qL50',
 'qL40',
 'qL30',
 'qL10',
 'uL137',
 'uL136',
 'uL130',
 'uL120',
 'uL110',
 'uL105',
 'uL100',
 'uL95',
 'uL90',
 'uL80',
 'uL70',
 'uL60',
 'uL50',
 'uL40',
 'uL30',
 'uL10',
 'vL137',
 'vL136',
 'vL130',
 'vL120',
 'vL110',
 'vL105',
 'vL100',
 'vL95',
 'vL90',
 'vL80',
 'vL70',
 'vL60',
 'vL50',
 'vL40',
 'vL30',
 'vL10',
 'tL137',
 'tL136',
 'tL130',
 'tL120',
 'tL110',
 'tL105',
 'tL100',
 'tL95',
 'tL90',
 'tL80',
 'tL70',
 'tL60',
 'tL50',
 'tL40',
 'tL30',
 'tL10',
 'sp',
 't2m',
 'z500',
 'u500',
 'q500',
 'v500',
 't500',]

    forecast_time = 24  # Example forecast time in hours
    ds = AiNWP6hDataSource(
        start_lat=20,
        end_lat=52,
        start_lon=-128, # This has converted to -180 to 180 format
        end_lon=-64,
        model='wxformer')
    res = ds(datetime.datetime(2020, 1, 1, 0),forecast_time)
    print(res)
    # print(res.isel(time=0).sel(channel='q2m').values)    