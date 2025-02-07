import numpy as np
from datetime import *
import math
import time, os, sys
import multiprocessing
import yaml
import copy
import xarray as xr
from functools import partial
from tqdm import tqdm
from utils.util import get_closest_gridbox, generate_date_list
from utils.util_pangu_infer import PanguUpscaler
import argparse

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Set up argument parser
parser = argparse.ArgumentParser(description="Load a YAML configuration file.")
parser.add_argument("config_path", type=str, help="Path to the YAML configuration file.")
parser.add_argument('--start_date', type=str, required=True, help='Start date in YYYYMMDD format')
parser.add_argument('--end_date', type=str, required=True, help='End date in YYYYMMDD format')
parser.add_argument('--ic', type=str, required=True, help='Initial condition')

# Parse arguments
args = parser.parse_args()

# Resolve absolute path
config_path = os.path.abspath(args.config_path)

# Check if the file exists
if not os.path.exists(config_path):
    print(f"Error: File not found at {config_path}")
else:
    # Load the configuration using the resolved path
    config = load_config(config_path)
    print("Configuration loaded successfully")

# Access the variables
MODEL = config['MODEL']
data_dir = config['data_dir']
output_rootdir = config['output_rootdir']
#start_input_date = config['start_input_date']
#end_date = config['end_date']
start_input_date = args.start_date
end_date = args.end_date
start_fcst_hour = config['start_fcst_hour']
end_fcst_hour = config['end_fcst_hour']
fcst_hour_interval = config['fcst_hour_interval']
# Access the upscaled model fields and derived fields
upscaled_model_fields = config['upscaled_model_fields']
upscaled_derived_fields = config['upscaled_derived_fields']

# intialize the variables
upscaled_combined_keys_empty = copy.deepcopy({**upscaled_model_fields, **upscaled_derived_fields})
fhr_list = list(range(start_fcst_hour, end_fcst_hour, fcst_hour_interval))
input_dates = generate_date_list(start_input_date, end_date)

print(input_dates)

basemap_products = get_closest_gridbox(model = MODEL, model_path='./nngridpts')
NN_GRID_PTS,interp_lons,interp_lats, in_lons_proj, in_lats_proj, x81, y81 = basemap_products
pangupscaler = PanguUpscaler(basemap_products=basemap_products)

def upscale_pangu_forecast(fcst_hr, input_date, ic):
    return pangupscaler.pangu_upscale_forecast(data_dir, 
                                       upscaled_model_fields, 
                                       upscaled_derived_fields, 
                                       input_date, 
                                       fcst_hr, ic)
    
for input_date in input_dates:
    dt_input_date = datetime.strptime(input_date, '%Y%m%d%H')
    print(f'Processing {dt_input_date}')
    # Create the output directory path for the netCDF file
    output_dir = f'{output_rootdir}/{dt_input_date.strftime("%Y%m%d%H")}/{args.ic}/'
    #output_filename = f'{output_dir}/{dt_input_date.strftime("%Y%m%d%H")}_{MODEL}_upscaled.nc'
    output_filename = f'{output_dir}/{dt_input_date.strftime("%Y%m%d%H")}_Pangu_{args.ic}_upscaled.nc'
    
    # Check if the output netCDF file exists
    if os.path.exists(output_filename):
        print(f'File already exists for {input_date}, skipping...')
        continue
    
    print(f'Running upscaling in parallel for {input_date}')

    nprocs = 6
    chunksize = int(math.ceil(len(fhr_list) / float(nprocs)))
    pool = multiprocessing.Pool(processes=nprocs)
    
    upscale_forecast_partial = partial(upscale_pangu_forecast, input_date=input_date, ic=args.ic)
    
    data = list(tqdm(pool.imap(upscale_forecast_partial, fhr_list, chunksize), 
                     total=len(fhr_list), 
                     desc="Upscaling Progress"))
    
    pool.close()
    pool.join()

    combined = {}
    for d in data:
        if isinstance(d, dict):
            combined.update(d)
        elif isinstance(d, tuple) and len(d) == 2:
            combined.update({d[0]: d[1]})
        
    upscaled_combined_keys = copy.deepcopy(upscaled_combined_keys_empty)
    for f in upscaled_combined_keys.keys():
        for fhr in fhr_list:
            if (len(combined[fhr][f]) > 0): 
                upscaled_combined_keys[f].append(combined[fhr][f])
            else: 
                print(f'null data for {f} at {fhr}hour')
                upscaled_combined_keys[f].append(np.ones((65,93))*np.nan)
                
    data_vars = {}
    for k, v in upscaled_combined_keys.items():
        data_vars[k] = (['fhr', 'y', 'x'], np.array(upscaled_combined_keys[k]).astype(np.float32))
    
    ds = xr.Dataset(data_vars=data_vars,
                    coords={'fhr': fhr_list, 'y': interp_lats[:,0], 'x': interp_lons[0,:]},
                    attrs={'init': dt_input_date.strftime('%Y%m%d%H')},
                    )
    comp = dict(zlib=True, complevel=1)
    encoding = {var: comp for var in ds.data_vars}

    os.makedirs(output_dir, exist_ok=True)
    
    ds.to_netcdf(output_filename, encoding=encoding)


