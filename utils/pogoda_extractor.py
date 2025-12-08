import xarray as xr
import pandas as pd
from pathlib import Path

# -----------------------------
# 0. Налаштування шляхів і років
# -----------------------------
base_dir = Path("Weather data")  # папка, в якій лежать підпапки 2023, 2024
years = [2023, 2024]
months = range(1, 13)

# -----------------------------
# 1. Точки для bidding zones
# -----------------------------
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
            print(f"[{year}-{month:02d}] Файл не знайдено: {grib_path}")
            continue

        print(f"\n=== Обробка {grib_path} ===")

        # -----------------------------
        # 2. Відкрити GRIB
        # -----------------------------
        ds = xr.open_dataset(
            grib_path,
            engine="cfgrib",
            backend_kwargs={"indexpath": ""}  # без створення *.idx файлів
        )
        print("Dataset loaded")

        # -----------------------------
        # 3. Витягнути часові ряди по зонах
        # -----------------------------
        zone_dataframes = []

        for zone, (lat, lon) in bidding_zone_points.items():
            print(f" -> {zone}: lat={lat}, lon={lon}")
            point = ds.sel(latitude=lat, longitude=lon, method="nearest")
            df_zone = point.to_dataframe().reset_index()
            df_zone["zone"] = zone
            zone_dataframes.append(df_zone)

        # -----------------------------
        # 4. Об’єднати зони в один DataFrame
        # -----------------------------
        df_all = pd.concat(zone_dataframes, ignore_index=True)

        cols_to_keep = ["valid_time", "zone"]
        for col in ["t2m", "u10", "v10", "msl", "tp", "ssrd", "u100", "v100"]:
            if col in df_all.columns:
                cols_to_keep.append(col)

        df_all = df_all[cols_to_keep]

        # Температура з Кельвінів в Цельсії
        if "t2m" in df_all.columns:
            df_all["t2m"] = df_all["t2m"] - 273.15

        # -----------------------------
        # 5. Зберегти у CSV в тій самій папці
        # -----------------------------
        csv_name = f"bidding_zone_weather_{year}_{month:02d}.csv"
        csv_path = year_dir / csv_name

        df_all.to_csv(csv_path, index=False)
        print(f"Збережено: {csv_path}")
        print(df_all.head())
        print(df_all.info())
