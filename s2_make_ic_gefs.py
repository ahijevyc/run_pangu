"""
Read input fields for PanguWeather, resize half-degree data to quarter-degree, and write to a GRIB file.

Input sources:
- AWS bucket (by running `~/bin/getGEFS.py` first)
- Craig Schwartz's HWT archive at `/glade/campaign/mmm/parc/schwartz/HWT2024/gefs/`
  - Available dates: 20230415-20230531 and 20240420-20240531.

Assumptions:
- Input files are in `$TMPDIR/noaa-gefs-pds`

TODO: Read from `/glade/campaign/mmm/parc/schwartz/HWT2024/gefs/`
"""

import multiprocessing
import os
from pathlib import Path
import sys
from typing import Dict, List

import earthkit.data as ekd
import pygrib
import tqdm
from skimage.transform import resize
from functools import partial

def process_member(time: str, mem: str) -> None:
    """Process a single ensemble member by reading, resizing, and writing GRIB data."""
    date = time[:8]
    hour = time[-2:]
    
    TMPDIR = Path(os.getenv("TMPDIR", "/tmp"))
    SCRATCH = Path(os.getenv("SCRATCH", "/scratch"))
    output = SCRATCH / "ai-models" / "input" / f"{date}{hour}" / mem / f"ge{mem}.t{hour}z.pgrb.0p25.f000"

    if os.path.exists(output):
        with pygrib.open(output) as grib:
            num_records = len(list(grib))
        if num_records == 69:
            print(f"{output} exists already. skipping")
            return
        else:
            print(f"{output} has only {num_records} records. Remake it.")
    os.makedirs(output.parent, exist_ok=True)

    kwargs = {"date": date, "time": hour, "step": "000", "year": date[:4]}

    # Load GRIB files
    pla = ekd.from_source("file", TMPDIR / "noaa-gefs-pds" / f"{date}{hour}/ge{mem}.t{hour}z.pgrb2a.0p50.f000", **kwargs)
    plb = ekd.from_source("file", TMPDIR / "noaa-gefs-pds" / f"{date}{hour}/ge{mem}.t{hour}z.pgrb2b.0p50.f000", **kwargs)
    sfc = ekd.from_source("file", TMPDIR / "noaa-gefs-pds" / f"{date}{hour}/ge{mem}.t{hour}z.pgrb2s.0p25.f000", **kwargs)

    param_sfc = ["prmsl", "10u", "10v", "2t"]
    param_pl = ["gh", "q", "t", "u", "v"]
    level_pl = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

    # Select and order fields
    fields_pl = (pla + plb).sel(param=param_pl, level=level_pl, levtype="pl").order_by(param=param_pl, level=level_pl)
    fields_sfc = sfc.sel(param=param_sfc, levtype="sfc").order_by(param=param_sfc)
    
    print(f"Processing {mem}: Writing to {output}")
    out = ekd.new_grib_output(output)

    G: Dict[str, float] = {"gh": 9.80665}  # Conversion factor for geopotential height
    PARAM: Dict[str, str] = {"gh": "z", "prmsl": "msl"}  # Parameter renaming
    
    reference_field = fields_sfc[0]
    grid_keys = [
        "Ni", "Nj", "latitudeOfFirstGridPoint", "longitudeOfFirstGridPoint",
        "latitudeOfLastGridPoint", "longitudeOfLastGridPoint",
        "iDirectionIncrement", "jDirectionIncrement", "gridType"
    ]
    
    for f in tqdm.tqdm(fields_pl + fields_sfc, desc=f"Processing {mem}"):
        param = f.metadata("shortName")
        grid_metadata = {key: reference_field[key] for key in grid_keys}
        arr = resize(f.to_numpy(), (721, 1440), order=0, mode='edge', anti_aliasing=False)
        
        out.write(
            arr * G.get(param, 1),
            template=f,
            centre=98,
            setLocalDefinition=1,
            subCentre=7,
            localDefinitionNumber=1,
            param=PARAM.get(param, param),
            packingType="grid_ccsds",  # Avoid missing FiniteDifferencingOrder key error
            **grid_metadata,
        )

def main(time: str) -> None:
    """Run the GRIB processing for all ensemble members in parallel."""
    members = ["c00"] + [f"p{i:02d}" for i in range(1, 31)]
    
    with multiprocessing.Pool(processes=8) as pool:
        pool.map(partial(process_member, time), members)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py YYYYMMDDHH")
        sys.exit(1)
    main(sys.argv[1])

