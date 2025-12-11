# Nordic renewable energy surplus analysis

## Goal
Build an hourly panel for Nordic bidding zones (DK1–DK2, FI, NO1–NO5, SE1–SE4) for 2023–2024, compute surplus flags, and summarise surplus events.

## Environment and credentials
- Create `.env/.env` and set `FINGRID_API_KEY=<your_key>` (get it from the Fingrid Open Data portal).
- Install dependencies via pip:
  - `pip install -r requirements.txt`
- Or create/activate the conda env:
  - `conda env create -f environment.yml`
  - `conda activate nordic-surplus`

## Data acquisition (API pulls)
Use `Pipeline_Data_API.ipynb` to download new raw data. It writes directly into the repo layout:
- Fingrid → `Dataset/fin/<year>/fi_<dataset>_<year>.csv`
- eSett → `Dataset/esett/<year>/*.csv`
- Energi Data Service → `Dataset/energi/<year>/<zone>_prices.csv`
- ERA5 → `Raw data/<year>/era5_weather_<year>_<month>.grib` and `Weather data/<year>/bidding_zone_weather_<year>_<month>.csv`

Toggle the `RUN_*` flags in the notebook before running to avoid unwanted downloads.

## Build the master panel
```bash
python build_surplus_dataset.py --years 2023 2024 --zones DK1 DK2 FI NO1 NO2 NO3 NO4 NO5 SE1 SE2 SE3 SE4 --output-dir processed
```
Outputs:
- `processed/master_panel.parquet`: hourly per-zone panel with prices, load, generation, flows, weather, balancing, surplus metrics.
- `processed/surplus_events.parquet`: grouped surplus events with start/end, duration, price and imbalance summaries.

## Analysis (surplus conditions & impacts)
```bash
python surplus_analysis.py
python surplus_plots.py
```
Results are written to `analysis_outputs/`:
- `surplus_vs_non.csv`, `correlations_net_surplus.csv`, `logit_surplus_odds.csv`, `top20_events_by_zone.csv`
- Plots: price vs surplus scatter, per-zone calendars, multi-zone calendars, per-zone event price paths.
