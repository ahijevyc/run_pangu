__all__ = ['WB2_Pangu_Datasource']

import dataclasses
from typing import List
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import fsspec

@dataclasses.dataclass
class WB2_Pangu_Datasource:
    start_lat: float
    end_lat: float
    start_lon: float
    end_lon: float
    fcst_hr_start: int
    fcst_hr_end: int
    levels: List[int]

    def __post_init__(self):
        url = "gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr"
        self.ds = xr.open_zarr(fsspec.get_mapper(url), chunks=None).isel(latitude=slice(None, None, -1))

    @property
    def time_means(self):
        raise NotImplementedError()

    def __call__(self, time: datetime, forecast_time: int):
        channels = self._get_channels(time, forecast_time)
        lat_slice = slice(self.start_lat, self.end_lat)
        lon_slice = slice(self.start_lon if self.start_lon >= 0 else self.start_lon % 360,
                          self.end_lon if self.end_lon >= 0 else self.end_lon % 360)
        
        result = channels.sel(
            latitude=lat_slice,
            longitude=lon_slice,
            level=self.levels
        )
        
        return result

    def _get_channels(self, time: datetime, forecast_time: int):        
        return self.ds.sel(time = time, prediction_timedelta=np.timedelta64(forecast_time, 'h'))

if __name__ == '__main__':
    weather_source = WB2_Pangu_Datasource(
        start_lat=20,
        end_lat=52,
        start_lon=-128,
        end_lon=-64,
        fcst_hr_start=48,
        fcst_hr_end=72,
        levels=[1000, 500]
    )
    
    result1 = weather_source(datetime(2022, 1, 1, 12, 0), forecast_time=48)
    result2 = weather_source(datetime(2022, 1, 1, 12, 0), forecast_time=72)
    print(result1,result2)