import dataclasses
from typing import List, Union
import datetime
from dateutil.relativedelta import relativedelta
import xarray as xr
import warnings
import concurrent.futures

# sfc 128
# pl 128

LEVEL_DATA_PATH = "/glade/campaign/collections/rda/data/d633000/e5.oper.an.pl/"
SFC_DATA_PATH = "/glade/campaign/collections/rda/data/d633000/e5.oper.an.sfc/"

var_to_file_name = {
    "t2m":"2t",
    "u10m":"10u",
    "v10m":"10v",
    "v100m":"100v",
    "u100m":"100u"
} 

CHANNEL_TO_CODE = {
    "z": 129, # z200
    "u": 131,
    "v": 132,
    "t": 130,
    "q": 133,
    "r": 157,
    "t2m": 167,
    "u10m": 165,
    "v10m": 166,
    # "u100m": 228246,
    # "v100m": 228247,
    "u100m": 246,
    "v100m": 247,

    "tcwv": 137,
    "sp": 134,
    "msl": 151,
    # total precip
    "tp": 228,
}


@dataclasses.dataclass
class PressureLevelCode:
    id: int
    name: str
    level: int = 0


@dataclasses.dataclass
class SingleLevelCode:
    id: int
    name: str
    code0: int = 128


def process_code(code, SFC_DATA_PATH, LEVEL_DATA_PATH, year, month, month_end_day, day, time):
    if code.name in ['u','v']:
        termll025 = 'll025uv'
    else:
        termll025 = 'll025sc'

    if isinstance(code, SingleLevelCode):
        path = f"{SFC_DATA_PATH}{year}{month}/e5.oper.an.sfc.{code.code0}_{code.id}_{code.name}.{termll025}.{year}{month}0100_{year}{month}{month_end_day}23.nc"
    elif isinstance(code, PressureLevelCode):
        path = f"{LEVEL_DATA_PATH}{year}{month}/e5.oper.an.pl.128_{code.id}_{code.name}.{termll025}.{year}{month}{day}00_{year}{month}{day}23.nc"
    else:
        raise TypeError("NO DATA TYPE FOUND.")

    path_data = xr.open_dataset(path)

    if list(path_data.keys())[0] != 'utc_date':
        var_name = list(path_data.keys())[0]
    else:
        var_name = list(path_data.keys())[1]
        warnings.warn(ResourceWarning(f"Please check var name {var_name}!"))

    if isinstance(code, SingleLevelCode):
        dataarray = path_data[var_name].loc[{"time": time}].expand_dims({"channel": 1})
    elif isinstance(code, PressureLevelCode):
        dataarray = path_data[var_name].loc[{"time": time, "level": code.level}].drop_vars("level").expand_dims({"channel": 1})

    dataarray = dataarray.rename({"latitude": "lat", "longitude": "lon"})
    return dataarray

def open_casper_nc(codes, time):
    # time
    year = str(time.year)
    month = str(time.month).zfill(2)
    day = str(time.day).zfill(2)
    hour = time.hour
    month_end_date = time + relativedelta(day=31)
    month_end_day = month_end_date.day
    # Main part
    dataarray_futures = []
    with concurrent.futures.ProcessPoolExecutor(8) as executor:
        for code in codes:
            # Pass additional required arguments to process_code
            future = executor.submit(process_code, code, SFC_DATA_PATH, LEVEL_DATA_PATH, year, month, month_end_day, day, time)
            dataarray_futures.append(future)

    dataarray_ls = [future.result() for future in dataarray_futures]
    dataarray_ls = xr.concat(dataarray_ls, dim="channel")
    return dataarray_ls    
    
def parse_channel(channel: str) -> Union[PressureLevelCode, SingleLevelCode]:
    if channel in list(var_to_file_name.keys()): 
        name = var_to_file_name[channel] 
    else :
        name = channel
    if channel in CHANNEL_TO_CODE:
        if channel in ['u100m','v100m','u10n','v10n','tcsw','tcrw','ltlt','lshf','lict','tp']:
            return SingleLevelCode(CHANNEL_TO_CODE[channel], name = name,code0=228)
        else:
            return SingleLevelCode(CHANNEL_TO_CODE[channel], name = name)
    else:
        code = CHANNEL_TO_CODE[channel[0]]
        name = name[0]
        level = int(channel[1:])
        return PressureLevelCode(code, name=name, level=int(level))

def _get_channels(time: datetime.datetime, channels: List[str]):
    codes = [parse_channel(c) for c in channels]
    # darray = _download_codes(client, codes, time)
    darray = open_casper_nc(codes, time)
    return (darray.assign_coords(channel=channels).assign_coords(time=time).expand_dims("time").transpose(
        "time", "channel", "lat", "lon")
            # .assign_coords(lon=darray["lon"] + 180.0)
            # .roll(lon=1440 // 2)
           )


@dataclasses.dataclass
class ERA5RDADataSource:
    channel_names: List[str]
    # client: Client = dataclasses.field(
    #     default_factory=lambda: Client(progress=False, quiet=False)
    # )

    @property
    def time_means(self):
        raise NotImplementedError()

    def __call__(self, time: datetime.datetime):
        return _get_channels(time, self.channel_names)


if __name__ == "__main__":

    pangu_channel = [
        'z1000', 'z925', 'z850', 'z700', 'z600', 'z500', 'z400', 'z300', 'z250', 'z200', 'z150', 'z100', 'z50', 'q1000',
        'q925', 'q850', 'q700', 'q600', 'q500', 'q400', 'q300', 'q250', 'q200', 'q150', 'q100', 'q50', 't1000', 't925',
        't850', 't700', 't600', 't500', 't400', 't300', 't250', 't200', 't150', 't100', 't50', 'u1000', 'u925', 'u850',
        'u700', 'u600', 'u500', 'u400', 'u300', 'u250', 'u200', 'u150', 'u100', 'u50', 'v1000', 'v925', 'v850', 'v700',
        'v600', 'v500', 'v400', 'v300', 'v250', 'v200', 'v150', 'v100', 'v50', 'msl', 'u10m', 'v10m', 't2m' #
    ]
    channel0 = ['tp'] #['t850', 'z1000', 'z700', 'z500', 'z300', 'tcwv', 't2m','tp']
    # for name in pangu_channel[-10:]:
    #     print(parse_channel(name))
    # for name in pangu_channel[:3]:
    ds = ERA5RDADataSource(pangu_channel)
    res = ds[datetime.datetime(2018, 1, 1, 0)]
    print(res)
    print(res.channel)
