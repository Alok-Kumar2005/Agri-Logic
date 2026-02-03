import xarray as xr
import numpy as np
import json
from pathlib import Path
from dask.diagnostics import ProgressBar


def process_era5_fast(input_path: Path, output_dir: Path):
    print(f"⚡ Hackathon ERA5 processing (CORRECT DASK): {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1️⃣ Open WITHOUT chunks (important)
    ds = xr.open_dataset(input_path)

    print("📊 Variables:", list(ds.data_vars))
    print("📐 Original shape:",
          ds.dims["valid_time"],
          ds.dims["latitude"],
          ds.dims["longitude"])

    # 2️⃣ Spatial downsampling FIRST (cheap, fast)
    print("🔻 Spatial downsampling...")
    ds = ds.isel(
        latitude=slice(None, None, 10),
        longitude=slice(None, None, 10)
    )

    print("📉 After spatial reduction:",
          ds.dims["valid_time"],
          ds.dims["latitude"],
          ds.dims["longitude"])

    # 3️⃣ NOW rechunk (aligned, efficient)
    ds = ds.chunk({
        "valid_time": 168,     # 1 week
        "latitude": -1,        # whole dim
        "longitude": -1
    })

    # 4️⃣ Temporal aggregation
    print("🕒 Daily resampling...")
    ds_daily = ds.resample(valid_time="1D").mean()

    # 5️⃣ Minimal stats (hackathon-appropriate)
    print("📈 Computing statistics...")
    stats = {}

    with ProgressBar():
        for var in ["u10", "v10", "t2m", "sp", "blh"]:
            if var in ds_daily:
                stats[var] = {
                    "mean": float(ds_daily[var].mean().compute())
                }

    # 6️⃣ Save output
    with open(output_dir / "era5_summary.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("✅ ERA5 processing COMPLETE (hackathon mode)")


if __name__ == "__main__":
    input_file = Path("data/raw/meteorological/data_stream.nc")
    output_dir = Path("data/processed/meteorological")

    process_era5_fast(input_file, output_dir)
