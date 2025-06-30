import os
import sys
import xarray as xr
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import shutil # For os.replace which is safer than remove/rename on some OS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s')

def process_single_file(input_file_path):
    """
    Function to process a single NetCDF file:
    - Checks size
    - Subsets data
    - Saves to a temporary file
    - Replaces original with subsetted file
    """
    logging.info(f"Starting processing for file: {input_file_path}")
    
    # Define temp output file name relative to the input file's directory
    # This prevents collisions if multiple processes write to 't.nc' in the same directory
    input_dir = os.path.dirname(input_file_path)
    input_filename = os.path.basename(input_file_path)
    temp_output_file_path = os.path.join(input_dir, f".{input_filename}.tmp_subset.nc") # Use a hidden temp file

    try:
        # Check file existence and size first
        if not os.path.exists(input_file_path):
            logging.warning(f"File not found, skipping: {input_file_path}")
            return input_file_path, "skipped_not_found"
            
        file_size = os.path.getsize(input_file_path)
        if file_size < 300_000_000: # Use underscore for readability
            logging.info(f"File {input_filename} is {file_size} bytes, less than 300MB. Skipping.")
            return input_file_path, "skipped_too_small"

        logging.info(f"Opening dataset: {input_file_path}")
        ds = xr.open_dataset(input_file_path)
        
        logging.info(f"Subsetting data for {input_filename}")
        # Select by integer position (index)
        selected_ds = ds.isel(channel=[0,1,2,3,9,11,13,14,15,24,26,27,28,35,37,39,40,41,48,50,52,53,54,61,63,65,66,67]) \
                          .sel(lat=slice(60,20), lon=slice(220,300))

        logging.info(f"Saving subsetted data to temporary file: {temp_output_file_path}")
        # Save to a temporary file
        selected_ds.to_netcdf(temp_output_file_path)
        
        # Close the original dataset AFTER writing the new one (important)
        ds.close()

        logging.info(f"Replacing original file {input_filename} with subsetted version.")
        # Atomically replace the original file with the new one
        try:
            os.replace(temp_output_file_path, input_file_path)
        except AttributeError: # os.replace not available in Python < 3.3
            os.remove(input_file_path)
            os.rename(temp_output_file_path, input_file_path)
            
        logging.info(f"Successfully processed: {input_file_path}")
        return input_file_path, "success"

    except Exception as e:
        logging.error(f"Error processing file {input_file_path}: {e}", exc_info=True)
        # Clean up temporary file if an error occurred
        if os.path.exists(temp_output_file_path):
            os.remove(temp_output_file_path)
        return input_file_path, f"error: {e}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python your_script_name.py <file1.nc> [file2.nc ...]")
        sys.exit(1) # Exit if no file paths are provided

    input_files = sys.argv[1:] # Get all file paths from command line arguments

    # Determine number of worker processes
    num_processes = os.cpu_count() if os.cpu_count() else 4 # Fallback if cpu_count is None
    logging.info(f"Using {num_processes} worker processes.")

    # Use ProcessPoolExecutor to run tasks in parallel
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit tasks to the pool
        future_to_file = {executor.submit(process_single_file, file_path): file_path for file_path in input_files}

        # Collect results as they complete
        for future in as_completed(future_to_file):
            file_path, status = future.result()
            logging.info(f"Finished processing {file_path} with status: {status}")

    logging.info("All files submitted and processing completed (or errors occurred).")
