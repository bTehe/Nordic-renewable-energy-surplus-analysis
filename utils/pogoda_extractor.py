import xarray as xr
import pandas as pd
from pathlib import Path

base_dir = Path("Weather data") 
years = [2023, 2024]
months = range(1, 13)

bidding_zone_points = {
    "DK1": (55.6, 9.2),
    "DK2": (55.7, 12.5),
    "NO1": (60.0, 10.0),
    "NO2": (59.0, 6.5),
    "NO3": (64.0, 11.0),
    "NO4": (69.0, 19.0),
    "NO5": (62.0, 5.5),
    "SE1": (66.0, 20.0),
    "SE2": (63.0, 17.0),
    "SE3": (59.5, 16.0),
    "SE4": (57.0, 15.0),
    "FI":  (61.5, 25.0),
}

for year in years:
    year_dir = base_dir / str(year)

    for month in months:
        grib_name = f"era5_weather_{year}_{month:02d}.grib"
        grib_path = year_dir / grib_name

        if not grib_path.exists():
            print(f"[{year}-{month:02d}] File not found: {grib_path}")
            continue

        print(f"\nProcessing {grib_path}")

        ds = xr.open_dataset(
            grib_path,
            engine="cfgrib",
            backend_kwargs={"indexpath": ""}
        )
        print("Dataset loaded")

        zone_dataframes = []

        for zone, (lat, lon) in bidding_zone_points.items():
            print(f" -> {zone}: lat={lat}, lon={lon}")
            point = ds.sel(latitude=lat, longitude=lon, method="nearest")
            df_zone = point.to_dataframe().reset_index()
            df_zone["zone"] = zone
            zone_dataframes.append(df_zone)

        df_all = pd.concat(zone_dataframes, ignore_index=True)

        cols_to_keep = ["valid_time", "zone"]
        for col in ["t2m", "u10", "v10", "msl", "tp", "ssrd", "u100", "v100"]:
            if col in df_all.columns:
                cols_to_keep.append(col)

        df_all = df_all[cols_to_keep]

        if "t2m" in df_all.columns:
            df_all["t2m"] = df_all["t2m"] - 273.15

        csv_name = f"bidding_zone_weather_{year}_{month:02d}.csv"
        csv_path = year_dir / csv_name

        df_all.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
        print(df_all.head())
        print(df_all.info())