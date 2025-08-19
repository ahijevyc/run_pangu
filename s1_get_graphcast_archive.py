#!/usr/bin/env python3
import os
import pandas as pd
import s3fs  # Using the s3fs package for direct S3 access

# Base S3 bucket URL for NOAA's GraphCast PDS
S3_BUCKET_BASE = "noaa-oar-mlwp-data" # Note: s3fs doesn't need the "s3://" prefix here
# Local directory where files will be saved
OUTPUT_BASE_DIR = "/glade/derecho/scratch/ahijevyc/ai-models/output/graphcast"

# --- Script ---

def download_graphcast_data(date, base_dir):
    """
    Downloads all GraphCast forecasts for a given date
    using the s3fs Python package.

    Args:
        date: The initialization time of the forecasts.
        base_dir (str): The local base directory to save data.
    """

    # Initialize the S3 file system object for anonymous access
    # This is the equivalent of the '--no-sign-request' CLI flag
    try:
        s3 = s3fs.S3FileSystem(anon=True)
    except Exception as e:
        print(f"Failed to initialize S3 connection: {e}")
        return

    print(f"\nProcessing {date}...")

    # Construct the source S3 path
    s3_source_path = f"{S3_BUCKET_BASE}/GRAP_v100_GFS/{date:%Y}/{date:%m%d}/GRAP_v100_GFS_{date:%Y%m%d%H}_f000_f240_06.nc"

    filename = os.path.basename(s3_source_path)

    # Construct the local destination path
    local_dest_dir = os.path.join(base_dir, f"{date:%Y%m%d%H}")

    full_local_path = os.path.join(local_dest_dir, filename)

    # Create the local directory if it doesn't exist
    try:
        os.makedirs(local_dest_dir, exist_ok=True)
        print(f"Ensured directory exists: {local_dest_dir}")
    except OSError as e:
        print(f"Error creating directory {local_dest_dir}: {e}")
        return

    if os.path.exists(full_local_path) and os.path.getsize(full_local_path) > 0:
        print(f"File '{full_local_path}' already exists and is not empty. Skipping.")
        return

    # Use s3fs.get() to download.
    try:
        print(f"{s3_source_path} to {local_dest_dir}...")
        s3.get(s3_source_path, local_dest_dir)
        print(f"Successfully synced data for {date}.")
    except Exception as e:
        print(f"Error downloading data for {date}: {e}")

    print("\n--- Download script finished. ---")

if __name__ == "__main__":
    dates2023 = pd.date_range(start="20230424", end="20230531")
    dates2024 = pd.date_range(start="20240420", end="20240531")
    for date in dates2023.union(dates2024):
        download_graphcast_data(date, OUTPUT_BASE_DIR)
