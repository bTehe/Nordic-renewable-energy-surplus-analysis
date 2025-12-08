import cdsapi

client = cdsapi.Client()

dataset = "reanalysis-era5-single-levels"

request = {
    "product_type": "reanalysis",
    "variable": [
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_sea_level_pressure",
        "total_precipitation",

        # Optional:
        "100m_u_component_of_wind",
        "100m_v_component_of_wind",
        "surface_solar_radiation_downwards"
    ],

    "year": ["2023"],
    "month": ["03"],
    "day": [f"{d:02d}" for d in range(1, 32)],
    "time": [f"{h:02d}:00" for h in range(24)],

    "area": [72, 5, 54, 32],

    "format": "grib"
}

client.retrieve(dataset, request).download("era5_weather_2023_03.grib")