# Processed datasets (outputs of `build_surplus_dataset.py`)

Location: `processed/`

## master_panel.parquet
Hourly panel with one row per `(datetime_utc, zone)` for 2023–2024.

Core keys
- `datetime_utc` : UTC timestamp (tz-naive)
- `zone` : bidding zone (DK1, DK2, FI, NO1–NO5, SE1–SE4)

Prices & load
- `da_price_eur_mwh` : day-ahead price (EUR/MWh)
- `actual_load_mw` : actual total load (MW)
- `load_forecast_mw` : day-ahead load forecast (MW)

Generation by type (MW)
- `gen_wind_onshore_mw`, `gen_wind_offshore_mw`
- `gen_hydro_ror_mw`, `gen_hydro_reservoir_mw`
- `gen_solar_mw`, `gen_nuclear_mw`, `gen_other_mw`
- `gen_total_mw` : sum of generation types

Cross-border flows & capacity (MW)
- `exports_mw`, `imports_mw`, `net_export_mw`
- `capacity_export_mw`, `capacity_import_mw`
- `export_utilisation`, `import_utilisation` : flows divided by capacity

Weather (from ERA5, per zone point)
- `t2m` (°C), `msl` (Pa)
- Wind components: `u10`, `v10`, `u100`, `v100`
- Wind speeds: `wind10`, `wind100` (m/s)

Balancing & volumes (eSett)
- `bal_price_down_eur_mwh`, `bal_price_up_eur_mwh`
- `esett_production_mwh`, `esett_consumption_mwh`
- `imbalance_mwh`, `abs_imbalance_mwh`

Calendar features
- `hour_of_day`, `weekday` (Mon=0), `month`, `year`, `season` (DJF/MAM/JJA/SON)

Surplus metrics
- `net_surplus_mw` : `gen_total_mw - actual_load_mw`
- `renewable_gen_mw` : wind + hydro + solar
- `renewable_share` : renewable_gen_mw / gen_total_mw
- `surplus_flag` : 1 if (net_surplus_mw > 0 & price ≤ 5) or price ≤ 0 else 0
- `z_net_surplus`, `z_da_price`, `z_net_export` : z-scores per zone
- `surplus_severity` : z_net_surplus − z_da_price + z_net_export

## surplus_events.parquet
Surplus episodes grouped by consecutive surplus hours per zone.

- `zone`
- `surplus_event_id` : incremental id within zone
- `start_time`, `end_time` (UTC)
- `duration_h` : number of surplus hours
- Prices: `min_price`, `mean_price`
- Surplus: `max_net_surplus_mw`
- Composition/conditions: `mean_renewable_share`, `mean_wind100`, `mean_load_mw`
- Flows: `mean_net_export_mw`, `max_export_utilisation`
- Balancing/imbalance: `mean_bal_price_down`, `mean_abs_imbalance`
- Severity: `mean_severity`