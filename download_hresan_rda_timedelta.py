import xarray as xr
import numpy as np
# from rda_gfs_fast import GFSDataSource
# from rda_era5 import ERA5RDADataSource
from rda_hresan import HRESANDataSource
import argparse
from pathlib import Path
import datetime
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Download and process data for a date range')
    parser.add_argument('--start_date', type=str, required=True, 
                        help='Start date in YYYYMMDD format')
    parser.add_argument('--end_date', type=str, required=True, 
                        help='End date in YYYYMMDD format')
    parser.add_argument('--output_dir', type=str, default='input_data',
                        help='Base output directory for saving files')
    parser.add_argument('--time_delta', type=int, default=0,
                        help='Time delta in hours. If -6, retrieve data 6 hours before 00UTC. Default is 0.')
    return parser.parse_args()

def generate_time_list(start_date, end_date):
    start = datetime.datetime.strptime(start_date, "%Y%m%d")
    end = datetime.datetime.strptime(end_date, "%Y%m%d")
    current = start
    time_list = []
    
    while current <= end:
        time_list.append(current)
        current += datetime.timedelta(days=1)
    
    return time_list

def ensure_directory_exists(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Ensure output directory exists
    ensure_directory_exists(args.output_dir)
    
    # Generate list of dates to process
    dates = generate_time_list(args.start_date, args.end_date)
    
    # Define Pangu channels

    # Define the pressure levels in reverse order
    pressure_levels = ['50', '100', '150', '200', '250', '300', '400', '500', '600', '700', '850', '925', '1000']
    
    # Define the variables that need to be flipped
    variables = ['z', 'q', 't', 'u', 'v']
    
    # Create the flipped part of the list
    flipped_channels = [f"{var}{level}" for var in variables for level in pressure_levels]
    
    # Define the surface variables
    surface_variables = ['msl', 'u10m', 'v10m', 't2m']
    
    # Combine the flipped channels with the surface variables
    pangu_channel = flipped_channels + surface_variables
    
    forecast_time = 0
    ds = HRESANDataSource(pangu_channel)
    
    # Process each date
    for date in dates:
        # Apply time delta if specified
        adjusted_date = date + datetime.timedelta(hours=args.time_delta)

        # Generate output filename
        output_file = f"pangu_hresan_init_{adjusted_date.strftime('%Y%m%d%H')}.nc"
        output_path = os.path.join(args.output_dir, output_file)
        
        # Skip if file already exists
        if os.path.exists(output_path):
            print(f"File {output_file} already exists, skipping...")
            continue
        
        # Process data
        try:
            res = ds[adjusted_date]
            res.to_netcdf(output_path)
            print(f"Successfully processed and saved data for {adjusted_date.strftime('%Y-%m-%d %H:%M UTC')}")
        except Exception as e:
            print(f"Error processing data for {adjusted_date.strftime('%Y-%m-%d %H:%M UTC')}: {str(e)}")

if __name__ == "__main__":
    main()