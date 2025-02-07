import os
import numpy as np
import onnx
import onnxruntime as ort
import xarray as xr
import argparse
from pathlib import Path
import datetime

# Define the coordinates
lat = np.linspace(90, -90, 721)
lon = np.linspace(0, 359.75, 1440)
upper_air_channels = [
    'z1000', 'z925', 'z850', 'z700', 'z600', 'z500', 'z400', 'z300', 'z250', 'z200', 'z150', 'z100', 'z50', 'q1000',
    'q925', 'q850', 'q700', 'q600', 'q500', 'q400', 'q300', 'q250', 'q200', 'q150', 'q100', 'q50', 't1000', 't925',
    't850', 't700', 't600', 't500', 't400', 't300', 't250', 't200', 't150', 't100', 't50', 'u1000', 'u925', 'u850',
    'u700', 'u600', 'u500', 'u400', 'u300', 'u250', 'u200', 'u150', 'u100', 'u50', 'v1000', 'v925', 'v850', 'v700',
    'v600', 'v500', 'v400', 'v300', 'v250', 'v200', 'v150', 'v100', 'v50'
]
surface_channels = ['msl', 'u10m', 'v10m', 't2m']


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run inference for a date range')
    parser.add_argument('--start_date', type=str, required=True, 
                        help='Start date in YYYYMMDD format')
    parser.add_argument('--end_date', type=str, required=True, 
                        help='End date in YYYYMMDD format')
    parser.add_argument('--inference_input_dir', type=str, required=True,
                        help='Directory containing input files')
    parser.add_argument('--inference_output_dir', type=str, required=True,
                        help='Directory for saving inference results')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory containing ONNX models')
    return parser.parse_args()

def generate_time_list(start_date, end_date):
    start = datetime.datetime.strptime(start_date, "%Y%m%d%H")
    end = datetime.datetime.strptime(end_date, "%Y%m%d%H")
    current = start
    time_list = []
    
    while current <= end:
        time_list.append(current)
        current += datetime.timedelta(days=1)
    
    return time_list

def ensure_directory_exists(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def setup_model_sessions(model_dir):
    # Load models
    model_24 = onnx.load(os.path.join(model_dir, 'pangu_weather_24.onnx'))
    model_6 = onnx.load(os.path.join(model_dir, 'pangu_weather_6.onnx'))

    # Set the behavior of onnxruntime
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    options.intra_op_num_threads = 4

    # Set the behavior of cuda provider
    cuda_provider_options = {'arena_extend_strategy': 'kSameAsRequested'}

    # Initialize sessions
    ort_session_24 = ort.InferenceSession(
        os.path.join(model_dir, 'pangu_weather_24.onnx'),
        sess_options=options,
        providers=[('CUDAExecutionProvider', cuda_provider_options)]
    )
    ort_session_6 = ort.InferenceSession(
        os.path.join(model_dir, 'pangu_weather_6.onnx'),
        sess_options=options,
        providers=[('CUDAExecutionProvider', cuda_provider_options)]
    )
    
    return ort_session_24, ort_session_6

def run_inference(date, input_dir, output_dir, ort_session_24, ort_session_6):
    # Define level lists
    # zlevels = list(reversed(['z1000', 'z925', 'z850', 'z700', 'z600', 'z500', 'z400', 'z300', 'z250', 'z200', 'z150', 'z100', 'z50']))
    # qlevels = list(reversed(['q1000', 'q925', 'q850', 'q700', 'q600', 'q500', 'q400', 'q300', 'q250', 'q200', 'q150', 'q100', 'q50']))
    # tlevels = list(reversed(['t1000', 't925', 't850', 't700', 't600', 't500', 't400', 't300', 't250', 't200', 't150', 't100', 't50']))
    # ulevels = list(reversed(['u1000', 'u925', 'u850', 'u700', 'u600', 'u500', 'u400', 'u300', 'u250', 'u200', 'u150', 'u100', 'u50']))
    # vlevels = list(reversed(['v1000', 'v925', 'v850', 'v700', 'v600', 'v500', 'v400', 'v300', 'v250', 'v200', 'v150', 'v100', 'v50']))
    zlevels = ['z1000', 'z925', 'z850', 'z700', 'z600', 'z500', 'z400', 'z300', 'z250', 'z200', 'z150', 'z100', 'z50',]
    qlevels = ['q1000','q925', 'q850', 'q700', 'q600', 'q500', 'q400', 'q300', 'q250', 'q200', 'q150', 'q100', 'q50',]
    tlevels = ['t1000', 't925','t850', 't700', 't600', 't500', 't400', 't300', 't250', 't200', 't150', 't100', 't50']
    ulevels = ['u1000', 'u925', 'u850','u700', 'u600', 'u500', 'u400', 'u300', 'u250', 'u200', 'u150', 'u100', 'u50',]
    vlevels = ['v1000', 'v925', 'v850', 'v700','v600', 'v500', 'v400', 'v300', 'v250', 'v200', 'v150', 'v100', 'v50',]
    surfaces = ['msl', 'u10m', 'v10m', 't2m']

    # Create output directory for this date
    date_output_dir = os.path.join(output_dir, date.strftime('%Y%m%d%H'))
    ensure_directory_exists(date_output_dir)

    # Check if all outputs already exist
    all_exists = True
    for i in range(40):
        nc_file = os.path.join(date_output_dir, f'pangu_hresan_pred_{str((i+1)*6).zfill(3)}.nc')
        if not os.path.exists(nc_file):
            all_exists = False
            break

    if all_exists:
        print(f"All outputs already exist for {date.strftime('%Y-%m-%d')}, skipping...")
        return

    # Load input data
    input_file = os.path.join(input_dir, f"pangu_hresan_init_{date.strftime('%Y%m%d%H')}.nc")
    ds_in = xr.open_dataarray(input_file)

    # Prepare input data
    input_upper = np.stack([
        ds_in.sel(channel=zlevels).to_numpy().squeeze(),
        ds_in.sel(channel=qlevels).to_numpy().squeeze(),
        ds_in.sel(channel=tlevels).to_numpy().squeeze(),
        ds_in.sel(channel=ulevels).to_numpy().squeeze(),
        ds_in.sel(channel=vlevels).to_numpy().squeeze()
    ], axis=0).astype(np.float32)
    input_surface = ds_in.sel(channel=surfaces).to_numpy().squeeze().astype(np.float32)

    # Modified inference loop
    input_24, input_surface_24 = input_upper, input_surface
    input, input_surface = input_upper, input_surface

    for i in range(40):
        print(f'Processing {date.strftime("%Y-%m-%d")} - {(i+1)*6} hour')
        
        if (i+1) % 4 == 0:
            output, output_surface = ort_session_24.run(None, {
                'input': input_24,
                'input_surface': input_surface_24
            })
            input_24, input_surface_24 = output, output_surface
        else:
            output, output_surface = ort_session_6.run(None, {
                'input': input,
                'input_surface': input_surface
            })
        
        input, input_surface = output, output_surface

        # Save results
        # np.save(os.path.join(date_output_dir, f'output_upper_{str((i+1)*6).zfill(3)}'), output)
        # np.save(os.path.join(date_output_dir, f'output_surface_{str((i+1)*6).zfill(3)}'), output_surface)
        
        # Create prediction timedelta
        pred_timedelta = np.timedelta64((i+1)*6, 'h').astype('timedelta64[ns]')
        
        # Reshape upper air output to combine variables and pressure levels
        output_reshaped = output.reshape(65, 721, 1440)  # 5 variables * 13 pressure levels = 65 channels
        
        # Create xarray DataArrays with proper dimensions
        da_upper = xr.DataArray(
            data=np.expand_dims(np.expand_dims(output_reshaped, axis=0), axis=0),
            coords={
                'init_time': [date],
                'prediction_timedelta': [pred_timedelta],
                'channel': upper_air_channels,
                'lat': lat,
                'lon': lon
            },
            dims=['init_time', 'prediction_timedelta', 'channel', 'lat', 'lon']
        )
        
        da_surface = xr.DataArray(
            data=np.expand_dims(np.expand_dims(output_surface, axis=0), axis=0),
            coords={
                'init_time': [date],
                'prediction_timedelta': [pred_timedelta],
                'channel': surface_channels,
                'lat': lat,
                'lon': lon
            },
            dims=['init_time', 'prediction_timedelta', 'channel', 'lat', 'lon']
        )
        
        # Combine upper air and surface data
        combined_channels = upper_air_channels + surface_channels
        combined_data = xr.concat([da_upper, da_surface], dim='channel').sel(lat=slice(60,20), lon=slice(220,300))
        
        # Save as netCDF
        output_filename = os.path.join(date_output_dir, f'pangu_hresan_pred_{str((i+1)*6).zfill(3)}.nc')
        combined_data.to_netcdf(output_filename)

def main():
    args = parse_arguments()
    
    # Ensure output directory exists
    ensure_directory_exists(args.inference_output_dir)
    
    # Setup model sessions
    ort_session_24, ort_session_6 = setup_model_sessions(args.model_dir)
    
    # Generate list of dates to process
    dates = generate_time_list(args.start_date, args.end_date)
    
    # Process each date
    for date in dates:
        try:
            run_inference(date, args.inference_input_dir, args.inference_output_dir,
                         ort_session_24, ort_session_6)
        except Exception as e:
            print(f"Error processing {date.strftime('%Y-%m-%d')}: {str(e)}")

if __name__ == "__main__":
    main()
