# Nordic renewable energy surplus analysis

## Goal
Build an hourly panel for Nordic bidding zones (DK1–DK2, FI, NO1–NO5, SE1–SE4) for 2023–2024, compute surplus flags, and summarise surplus events.

## Prerequisites
- Python 3.10+ with `pandas` and `numpy` installed (`pip install pandas numpy pyarrow`).
- Local data folders present:
  - `Dataset/` (ENTSO-E, eSett, Energi, etc.)
  - `Weather data/` (monthly bidding_zone_weather_YYYY_MM.csv)

## How to run
```bash
python build_surplus_dataset.py --years 2023 2024 --zones DK1 DK2 FI NO1 NO2 NO3 NO4 NO5 SE1 SE2 SE3 SE4 --output-dir processed
```

Flags:
- `--years`: one or more years to include (default: 2023 2024)
- `--zones`: zone codes to include (default: all Nordic bidding zones)
- `--output-dir`: folder for outputs (default: `processed`)

## Outputs
- `processed/master_panel.parquet`: hourly per-zone panel with prices, load, generation, flows, weather, balancing, surplus metrics.
- `processed/surplus_events.parquet`: grouped surplus events with start/end, duration, price and imbalance summaries.

## Analysis (surplus conditions & impacts)
Run the lightweight analysis to replicate the project steps (surplus vs non-surplus contrasts, correlations, logistic odds ratios, ranked events):
```bash
python surplus_analysis.py
```
Results are written to `analysis_outputs/`:
- `surplus_vs_non.csv`: hourly contrasts (prices, surplus, exports, utilisation, balancing).
- `correlations_net_surplus.csv`: Pearson r and p-values vs key drivers per zone.
- `logit_surplus_odds.csv`: odds ratios from logistic regression of surplus_flag on load, renewables, wind, temperature, net exports.
- `top20_events_by_zone.csv`: ranked surplus events (mean severity) with timing and key metrics.
