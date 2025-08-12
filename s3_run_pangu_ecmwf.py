import os
import numpy as np
import onnx
import onnxruntime as ort
import xarray as xr
import argparse
from pathlib import Path
import pandas as pd

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
                        help='Start date in YYYYMMDDHH format')
    parser.add_argument('--end_date', type=str, required=True,
                        help='End date in YYYYMMDDHH format')
    parser.add_argument('--inference_input_dir', type=str, required=True,
                        help='Directory containing input files')
    parser.add_argument('--inference_output_dir', type=str, required=True,
                        help='Directory for saving inference results')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory containing ONNX models')
    parser.add_argument('--ic', type=str, required=True,
                        help='Initial condition identifier')
    return parser.parse_args()


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

def run_inference(ds_in: xr.DataArray, ort_session_24, ort_session_6, fhr_end: int):
    """
    Runs the Pangu-Weather inference loop for a given initial condition dataset.
    
    Args:
        ds_in (xr.DataArray): The input DataArray containing the initial conditions.
        ort_session_24 (ort.InferenceSession): The ONNX session for the 24-hour model.
        ort_session_6 (ort.InferenceSession): The ONNX session for the 6-hour model.
        fhr_end (int): run to this forecast hour
    """
    # Define level lists
    zlevels = ['z1000', 'z925', 'z850', 'z700', 'z600', 'z500', 'z400', 'z300', 'z250', 'z200', 'z150', 'z100', 'z50']
    qlevels = ['q1000', 'q925', 'q850', 'q700', 'q600', 'q500', 'q400', 'q300', 'q250', 'q200', 'q150', 'q100', 'q50']
    tlevels = ['t1000', 't925', 't850', 't700', 't600', 't500', 't400', 't300', 't250', 't200', 't150', 't100', 't50']
    ulevels = ['u1000', 'u925', 'u850', 'u700', 'u600', 'u500', 'u400', 'u300', 'u250', 'u200', 'u150', 'u100', 'u50']
    vlevels = ['v1000', 'v925', 'v850', 'v700', 'v600', 'v500', 'v400', 'v300', 'v250', 'v200', 'v150', 'v100', 'v50']
    surfaces = ['msl', 'u10m', 'v10m', 't2m']

    date = ds_in.time.data

    # Prepare input data from the provided xarray.DataArray
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

    inferences=[]
    for i in range(fhr_end//6):
        print(f'Processing {date} - {(i+1)*6} hour')

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

        # Create prediction timedelta
        pred_timedelta = pd.to_timedelta((i+1)*6, unit='h')

        # Reshape upper air output to combine variables and pressure levels
        output_reshaped = output.reshape(65, 721, 1440)  # 5 variables * 13 pressure levels = 65 channels

        # Create xarray DataArrays with proper dimensions
        da_upper = xr.DataArray(
            data=np.expand_dims(np.expand_dims(output_reshaped, axis=0), axis=0),
            coords={
                'init_time': date,
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
                'init_time': date,
                'prediction_timedelta': [pred_timedelta],
                'channel': surface_channels,
                'lat': lat,
                'lon': lon
            },
            dims=['init_time', 'prediction_timedelta', 'channel', 'lat', 'lon']
        )

        # Combine upper air and surface data
        combined_data = xr.concat([da_upper, da_surface], dim='channel').sel(lat=slice(60, 20), lon=slice(220, 300))
    
        inferences.append(combined_data)
    return inferences

def main():
    args = parse_arguments()

    # Ensure output directory exists
    ensure_directory_exists(args.inference_output_dir)

    # Setup model sessions
    ort_session_24, ort_session_6 = setup_model_sessions(args.model_dir)

    # Generate list of dates to process
    dates = pd.date_range(args.start_date, args.end_date)

    date_output_dir = args.inference_output_dir
    ic = args.ic
    fhr_end = 240
    # Process each date
    for date in dates:
        # Create output directory
        ensure_directory_exists(date_output_dir)

        # Check if all outputs already exist
        all_exists = True
        for fhr in range(6, 241, 6):
            nc_file = os.path.join(date_output_dir, f'pangu_{ic}_pred_{fhr:03d}.nc')
            if not os.path.exists(nc_file):
                all_exists = False
                break

        if all_exists:
            print(f"All outputs already exist for {date}, skipping...")
            continue

        try:
            # --- I/O logic is now here, in the main loop ---
            # Construct the input file path
            input_file = os.path.join(args.inference_input_dir, f"pangu_{ic}_init_{date.strftime('%Y%m%d%H')}.nc")
            
            # Open the dataset
            ds_in = xr.open_dataarray(input_file)

            # --- Pass the dataset object to the inference function ---
            inferences = run_inference(ds_in, ort_session_24, ort_session_6, fhr_end)

            for combined_data in inferences: 
                # Save as netCDF
                fhr = combined_data.prediction_timedelta.squeeze() / pd.to_timedelta('1h')
                output_filename = os.path.join(date_output_dir, f'pangu_{ic}_pred_{fhr:03.0f}.nc')
                combined_data.to_netcdf(output_filename)

        except FileNotFoundError:
            print(f"Input file not found for {date.strftime('%Y-%m-%d %H')}: {input_file}")
        except Exception as e:
            print(f"Error processing {date.strftime('%Y-%m-%d %H')}: {str(e)}")

if __name__ == "__main__":
    main()
