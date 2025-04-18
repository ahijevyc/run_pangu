import logging
import os
from pathlib import Path
import s3fs
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

TMPDIR = Path(os.getenv("TMPDIR"))
bucket = "noaa-gefs-pds"
fs = s3fs.S3FileSystem(anon=True)
members = [f"p{i:02d}" for i in range(1, 31)] + ["c00"]


def download_file(fs, s3_path, save_path):
    """Download a file from S3 using s3fs and save it locally."""
    if os.path.exists(save_path):
        logging.info(f"{save_path} exists")
        return
    try:
        fs.get(s3_path, save_path)
        print(f"Downloaded: {save_path}")
    except Exception as e:
        print(f"Failed to download {s3_path}: {e}")


def download_time(date: pd.Timestamp):
    yyyymmdd = date.strftime("%Y%m%d")
    hh = date.strftime("%H")
    date_save_dir = os.path.join(TMPDIR / bucket / date.strftime("%Y%m%d%H"))
    os.makedirs(date_save_dir, exist_ok=True)
    
    base_s3_path = f"s3://{bucket}/gefs.{yyyymmdd}/{hh}/atmos/"
    s3_paths = [(fs, base_s3_path + f"pgrb2sp25/ge{mem}.t{hh}z.pgrb2s.0p25.f000", os.path.join(date_save_dir, f"ge{mem}.t{hh}z.pgrb2s.0p25.f000")) for mem in members]
    s3_paths += [(fs, base_s3_path + f"pgrb2ap5/ge{mem}.t{hh}z.pgrb2a.0p50.f000", os.path.join(date_save_dir, f"ge{mem}.t{hh}z.pgrb2a.0p50.f000")) for mem in members]
    s3_paths += [(fs, base_s3_path + f"pgrb2bp5/ge{mem}.t{hh}z.pgrb2b.0p50.f000", os.path.join(date_save_dir, f"ge{mem}.t{hh}z.pgrb2b.0p50.f000")) for mem in members]
    
    # Use ThreadPoolExecutor for concurrent downloads
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(lambda args: download_file(*args), s3_paths)


def main():
    date_range = pd.date_range(start="2023-04-15", end="2023-05-31", freq="D")
    date_range = date_range.union(
                 pd.date_range(start="2024-04-20", end="2024-05-31", freq="D")
    )
    
    for date in date_range:
        download_time(date)

if __name__ == "__main__":
    main()

