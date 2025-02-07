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

# FengWu channel definitions
surface_channels = ['u10m', 'v10m', 't2m', 'msl']
variables = ['z', 'q', 'u', 'v', 't']
pressure_levels = ['50', '100', '150', '200', '250', '300', '400', '500', '600', '700', '850', '925', '1000']
fengwu_channels = surface_channels + [f"{var}{level}" for var in variables for level in pressure_levels]

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run inference for a date range with FengWu')
    parser.add_argument('--start_date', type=str, required=True,
                        help='Start date in YYYYMMDD format')
    parser.add_argument('--end_date', type=str, required=True,
                        help='End date in YYYYMMDD format')
    parser.add_argument('--inference_input_dir', type=str, required=True,
                        help='Directory containing input files for current time')
    parser.add_argument('--inference_input_dir_minus_6', type=str, required=True,
                        help='Directory containing input files for 6 hours prior')
    parser.add_argument('--inference_output_dir', type=str, required=True,
                        help='Directory for saving inference results')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory containing ONNX models')
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


def setup_model_sessions(model_dir):
    # Load model
    model_6 = onnx.load(os.path.join(model_dir, 'fengwu_v1.onnx'))

    # Set the behavior of onnxruntime
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    options.intra_op_num_threads = 4

    # Set the behavior of cuda provider
    cuda_provider_options = {'arena_extend_strategy': 'kSameAsRequested'}

    # Initialize session
    ort_session_6 = ort.InferenceSession(
        os.path.join(model_dir, 'fengwu_v1.onnx'),
        sess_options=options,
        providers=[('CUDAExecutionProvider', cuda_provider_options)]
    )
    return ort_session_6

# def prepare_fengwu_input(date, input_dir, input_dir_minus_6):
#     # Load input data for current time
#     input_file = os.path.join(input_dir, f"pangu_hresan_init_{date.strftime('%Y%m%d%H')}.nc")
#     ds_in = xr.open_dataarray(input_file)
    
#     # Load input data for 6 hours prior
#     date_minus_6 = date - datetime.timedelta(hours=6)
#     input_file_minus_6 = os.path.join(input_dir_minus_6, f"pangu_hresan_init_{date_minus_6.strftime('%Y%m%d%H')}.nc")
#     ds_in_minus_6 = xr.open_dataarray(input_file_minus_6)
    
#     # Stack data into the format expected by FengWu [69, 721, 1440]
#     input1 = np.stack([ds_in.sel(channel=surface_channels).to_numpy().squeeze()] +
#                      [ds_in.sel(channel=f"{var}{level}").to_numpy().squeeze() for var in variables for level in pressure_levels],
#                      axis=0).astype(np.float32)
#     input2 = np.stack([ds_in_minus_6.sel(channel=surface_channels).to_numpy().squeeze()] +
#                      [ds_in_minus_6.sel(channel=f"{var}{level}").to_numpy().squeeze() for var in variables for level in pressure_levels],
#                      axis=0).astype(np.float32)
#     print('inputs shape:', input1.shape, input2.shape)
#     return input1, input2


def prepare_fengwu_input(date, input_dir, input_dir_minus_6):
    # Load input data for current time
    input_file = os.path.join(input_dir, f"pangu_hresan_init_{date.strftime('%Y%m%d%H')}.nc")
    try:
        ds_in = xr.open_dataarray(input_file)
    except FileNotFoundError:
         raise FileNotFoundError(f"Input file not found: {input_file}")

    # Load input data for 6 hours prior
    date_minus_6 = date - datetime.timedelta(hours=6)
    input_file_minus_6 = os.path.join(input_dir_minus_6, f"pangu_hresan_init_{date_minus_6.strftime('%Y%m%d%H')}.nc")
    try:
         ds_in_minus_6 = xr.open_dataarray(input_file_minus_6)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_file_minus_6}")

    # Prepare input1
    input1_list = []
    
    # Surface channels
    for sfc_channel in surface_channels:
        surface_data = ds_in.sel(channel=sfc_channel).to_numpy().squeeze()
        input1_list.append(surface_data)
        print(f"Shape of surface_data: {surface_data.shape}")
    

    # Upper-air channels
    for var in variables:
        for level in pressure_levels:
            var_level_data = ds_in.sel(channel=f"{var}{level}").to_numpy().squeeze()
            print(f"Shape of {var}{level}: {var_level_data.shape}")
            input1_list.append(var_level_data)
    
    input1 = np.stack(input1_list, axis=0).astype(np.float32)
    print(f"Shape of input1 after stacking: {input1.shape}")

    # Prepare input2
    input2_list = []

    # Surface channels
    for sfc_channel in surface_channels:
        surface_data_minus_6 = ds_in_minus_6.sel(channel=sfc_channel).to_numpy().squeeze()
        input2_list.append(surface_data_minus_6)
        print(f"Shape of surface_data_minus_6: {surface_data_minus_6.shape}")
    
    # Upper-air channels
    for var in variables:
        for level in pressure_levels:
            var_level_data_minus_6 = ds_in_minus_6.sel(channel=f"{var}{level}").to_numpy().squeeze()
            print(f"Shape of {var}{level} in input2: {var_level_data_minus_6.shape}")
            input2_list.append(var_level_data_minus_6)

    input2 = np.stack(input2_list, axis=0).astype(np.float32)
    print(f"Shape of input2 after stacking: {input2.shape}")

    return input1, input2

def run_inference(date, input_dir, input_dir_minus_6, output_dir, ort_session_6, data_mean, data_std):

    # Create output directory for this date
    date_output_dir = os.path.join(output_dir, date.strftime('%Y%m%d%H'))
    ensure_directory_exists(date_output_dir)

    # Check if all outputs already exist
    all_exists = True
    for i in range(56):
        nc_file = os.path.join(date_output_dir, f'fengwu_pred_{str((i+1)*6).zfill(3)}.nc')
        if not os.path.exists(nc_file):
            all_exists = False
            break

    if all_exists:
        print(f"All outputs already exist for {date.strftime('%Y-%m-%d')}, skipping...")
        return
    
    # Prepare input data
    input_current, input_prior = prepare_fengwu_input(date, input_dir, input_dir_minus_6)

    # Normalize input data
    input_current_after_norm = (input_current - data_mean) / data_std
    input_prior_after_norm = (input_prior - data_mean) / data_std
    input_fengwu = np.concatenate((input_prior_after_norm, input_current_after_norm), axis=0)[np.newaxis, :, :, :]
    input_fengwu = input_fengwu.astype(np.float32)
    # print(input_fengwu.shape)
    
    input = input_fengwu

    for i in range(40):
        print(f'Processing {date.strftime("%Y-%m-%d")} - {(i+1)*6} hour')
        output = ort_session_6.run(None, {'input':input})[0]
        input = np.concatenate((input[:, 69:], output[:, :69]), axis=1)
        output = (output[0, :69] * data_std) + data_mean

        # Create prediction timedelta
        pred_timedelta = np.timedelta64((i+1)*6, 'h').astype('timedelta64[ns]')
        
        # Create xarray DataArrays with proper dimensions
        da_output = xr.DataArray(
            data=np.expand_dims(np.expand_dims(output, axis=0), axis=0),
            coords={
                'init_time': [date],
                'prediction_timedelta': [pred_timedelta],
                'channel': fengwu_channels,
                'lat': lat,
                'lon': lon
            },
            dims=['init_time', 'prediction_timedelta', 'channel', 'lat', 'lon']
        ).sel(lat=slice(60,20), lon=slice(220,300))

        # Save as netCDF
        output_filename = os.path.join(date_output_dir, f'fengwu_pred_{str((i+1)*6).zfill(3)}.nc')
        da_output.to_netcdf(output_filename)


def main():
    args = parse_arguments()
    
    # Ensure output directory exists
    ensure_directory_exists(args.inference_output_dir)
    
    # Setup model sessions
    ort_session_6 = setup_model_sessions(args.model_dir)
    
    # Load normalization data
    data_mean = np.load(os.path.join(args.model_dir, "data_mean.npy"))[:, np.newaxis, np.newaxis]
    data_std = np.load(os.path.join(args.model_dir, "data_std.npy"))[:, np.newaxis, np.newaxis]

    # Generate list of dates to process
    dates = generate_time_list(args.start_date, args.end_date)
    
    # Process each date
    for date in dates:
        try:
            run_inference(date, args.inference_input_dir, args.inference_input_dir_minus_6, 
                         args.inference_output_dir, ort_session_6, data_mean, data_std)
        except Exception as e:
            print(f"Error processing {date.strftime('%Y-%m-%d')}: {str(e)}")


if __name__ == "__main__":
    main()