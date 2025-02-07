import xarray as xr
import numpy as np

ds = xr.open_dataarray('pangu_hres_input_data/pangu_hresan_init_2025012000.nc')
#ds = xr.open_dataset('pangu_hresan_input_data/pangu_hresan_init_2024050100.nc')

field = 'u'
levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000] 
channels = [ field+str(l) for l in levels ]
print(channels)

for ch in channels:
    test = ds.sel(channel=ch)
    print(test.shape, test.values.min(), test.values.max())
