"""
Build an hourly master panel and surplus events for Nordic bidding zones.

Inputs: local `Dataset` and `Weather data` folders already extracted.
Outputs (default `processed/`):
  - master_panel.parquet : hourly per-zone features
  - surplus_events.parquet : surplus episode summaries
"""
from __future__ import annotations

import argparse
from functools import reduce
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


TZ_SOURCE = "Europe/Brussels"
DEFAULT_ZONES = [
    "DK1",
    "DK2",
    "FI",
    "NO1",
    "NO2",
    "NO3",
    "NO4",
    "NO5",
    "SE1",
    "SE2",
    "SE3",
    "SE4",
]
DEFAULT_YEARS = [2023, 2024]


def parse_mtu_to_utc(series: pd.Series) -> pd.Series:
    """Parse ENTSO-E MTU column (CET/CEST annotated) to UTC start timestamp."""
    s = series.astype(str)
    start = s.str.split(" - ").str[0]
    tz_token = start.str.extract(r"\((CEST|CET)\)", expand=False)
    clean = start.str.replace(r"\s*\((?:CEST|CET)\)", "", regex=True).str.strip()

    dt = pd.to_datetime(clean, dayfirst=True, errors="coerce")
    res = dt.copy()

    mask_cet = tz_token == "CET"
    mask_cest = tz_token == "CEST"
    res[mask_cet] = dt[mask_cet] - pd.Timedelta(hours=1)
    res[mask_cest] = dt[mask_cest] - pd.Timedelta(hours=2)

    mask_other = ~(mask_cet | mask_cest)
    if mask_other.any():
        res.loc[mask_other] = (
            dt[mask_other]
            .dt.tz_localize(TZ_SOURCE, ambiguous="NaT", nonexistent="shift_forward")
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
        )

    return res


def clean_zone(value: str | float) -> str | None:
    """Strip BZN| prefix and return zone code."""
    if pd.isna(value):
        return None
    return str(value).split("|")[-1]


def find_mtu_column(columns: Sequence[str]) -> str:
    for col in columns:
        if "MTU" in col:
            return col
    raise ValueError("MTU column not found")


def merge_on_datetime_zone(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=["datetime_utc", "zone"])
    return reduce(
        lambda left, right: pd.merge(
            left, right, how="outer", on=["datetime_utc", "zone"]
        ),
        frames,
    )


def load_entsoe_prices(entsoe_dir: Path, years: List[int], zones: List[str]) -> pd.DataFrame:
    records = []
    for year in years:
        for zone in zones:
            path = next((entsoe_dir / str(year) / zone).glob("GUI_ENERGY_PRICES_*.csv"), None)
            if not path:
                continue
            df = pd.read_csv(path)
            time_col = find_mtu_column(df.columns)
            df["datetime_utc"] = parse_mtu_to_utc(df[time_col])
            df["zone"] = zone
            df = df.rename(
                columns={
                    "Day-ahead Price (EUR/MWh)": "da_price_eur_mwh",
                }
            )
            records.append(
                df[
                    [
                        "datetime_utc",
                        "zone",
                        "da_price_eur_mwh",
                    ]
                ]
            )
    if not records:
        return pd.DataFrame(columns=["datetime_utc", "zone", "da_price_eur_mwh"])
    return pd.concat(records, ignore_index=True)


def load_entsoe_load(entsoe_dir: Path, years: List[int], zones: List[str]) -> pd.DataFrame:
    records = []
    for year in years:
        for zone in zones:
            path = next((entsoe_dir / str(year) / zone).glob("GUI_TOTAL_LOAD_DAYAHEAD_*.csv"), None)
            if not path:
                continue
            df = pd.read_csv(path)
            time_col = find_mtu_column(df.columns)
            df["datetime_utc"] = parse_mtu_to_utc(df[time_col])
            df["zone"] = zone
            df = df.rename(
                columns={
                    "Actual Total Load (MW)": "actual_load_mw",
                    "Day-ahead Total Load Forecast (MW)": "load_forecast_mw",
                }
            )
            df["actual_load_mw"] = pd.to_numeric(df["actual_load_mw"], errors="coerce")
            df["load_forecast_mw"] = pd.to_numeric(df["load_forecast_mw"], errors="coerce")
            records.append(df[["datetime_utc", "zone", "actual_load_mw", "load_forecast_mw"]])
    if not records:
        return pd.DataFrame(columns=["datetime_utc", "zone", "actual_load_mw", "load_forecast_mw"])
    return pd.concat(records, ignore_index=True)


def load_entsoe_generation(entsoe_dir: Path, years: List[int], zones: List[str]) -> pd.DataFrame:
    records = []
    for year in years:
        for zone in zones:
            path = next(
                (entsoe_dir / str(year) / zone).glob("AGGREGATED_GENERATION_PER_TYPE_GENERATION_*.csv"), None
            )
            if not path:
                continue
            df = pd.read_csv(path)
            time_col = find_mtu_column(df.columns)
            df["datetime_utc"] = parse_mtu_to_utc(df[time_col])
            df["zone"] = zone
            df["prod_type_clean"] = (
                df["Production Type"]
                .str.lower()
                .str.replace("[^a-z0-9]+", "_", regex=True)
                .str.strip("_")
            )
            df["generation_mw"] = pd.to_numeric(df["Generation (MW)"], errors="coerce")
            records.append(df[["datetime_utc", "zone", "prod_type_clean", "generation_mw"]])
    if not records:
        return pd.DataFrame(columns=["datetime_utc", "zone"])

    long_df = pd.concat(records, ignore_index=True)
    pivot = (
        long_df.pivot_table(
            index=["datetime_utc", "zone"],
            columns="prod_type_clean",
            values="generation_mw",
            aggfunc="sum",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    raw_cols = [c for c in pivot.columns if c not in {"datetime_utc", "zone"}]
    pivot["gen_total_mw"] = pivot[raw_cols].sum(axis=1, min_count=1)

    mapping = {
        "wind_onshore": "gen_wind_onshore_mw",
        "wind_offshore": "gen_wind_offshore_mw",
        "hydro_run_of_river_and_poundage": "gen_hydro_ror_mw",
        "hydro_water_reservoir": "gen_hydro_reservoir_mw",
        "solar": "gen_solar_mw",
        "nuclear": "gen_nuclear_mw",
    }

    for raw_name, out_name in mapping.items():
        pivot[out_name] = pivot[raw_name] if raw_name in pivot else 0.0

    pivot["gen_other_mw"] = pivot["gen_total_mw"] - pivot[list(mapping.values())].sum(axis=1, min_count=0)

    keep_cols = ["datetime_utc", "zone"] + list(mapping.values()) + ["gen_other_mw", "gen_total_mw"]
    return pivot[keep_cols]


def load_flow_edges(entsoe_dir: Path, years: List[int], zones: List[str]) -> pd.DataFrame:
    records = []
    for year in years:
        for zone in zones:
            path = next(
                (entsoe_dir / str(year) / zone).glob("GUI_NET_CROSS_BORDER_PHYSICAL_FLOWS_*.csv"), None
            )
            if not path:
                continue
            df = pd.read_csv(path)
            time_col = find_mtu_column(df.columns)
            df["datetime_utc"] = parse_mtu_to_utc(df[time_col])
            df["out_zone"] = df["Out Area"].apply(clean_zone)
            df["in_zone"] = df["In Area"].apply(clean_zone)
            df["flow_mw"] = pd.to_numeric(df["Physical Flow (MW)"], errors="coerce")
            records.append(df[["datetime_utc", "out_zone", "in_zone", "flow_mw"]])
    if not records:
        return pd.DataFrame(columns=["datetime_utc", "out_zone", "in_zone", "flow_mw"])
    edges = pd.concat(records, ignore_index=True)
    edges = edges.dropna(subset=["datetime_utc", "out_zone", "in_zone", "flow_mw"]).drop_duplicates()
    return edges


def load_capacity_edges(entsoe_dir: Path, years: List[int], zones: List[str]) -> pd.DataFrame:
    records = []
    for year in years:
        for zone in zones:
            path = next(
                (entsoe_dir / str(year) / zone).glob("GUI_FORECASTED_TRANSFER_CAPACITIES_DAY-AHEAD_*.csv"), None
            )
            if not path:
                continue
            df = pd.read_csv(path)
            time_col = find_mtu_column(df.columns)
            df["datetime_utc"] = parse_mtu_to_utc(df[time_col])
            df["out_zone"] = df["Out Area"].apply(clean_zone)
            df["in_zone"] = df["In Area"].apply(clean_zone)
            df["capacity_mw"] = pd.to_numeric(df["Capacity (MW)"], errors="coerce")
            records.append(df[["datetime_utc", "out_zone", "in_zone", "capacity_mw"]])
    if not records:
        return pd.DataFrame(columns=["datetime_utc", "out_zone", "in_zone", "capacity_mw"])
    edges = pd.concat(records, ignore_index=True)
    return edges.dropna(subset=["datetime_utc", "out_zone", "in_zone", "capacity_mw"]).drop_duplicates()


def aggregate_zone_flows(flow_edges: pd.DataFrame) -> pd.DataFrame:
    if flow_edges.empty:
        return pd.DataFrame(columns=["datetime_utc", "zone", "exports_mw", "imports_mw", "net_export_mw"])

    out_view = flow_edges[["datetime_utc", "out_zone", "flow_mw"]].rename(columns={"out_zone": "zone"})
    out_view = out_view.assign(
        exports_mw=out_view["flow_mw"].clip(lower=0),
        imports_mw=(-out_view["flow_mw"]).clip(lower=0),
    )

    in_view = flow_edges[["datetime_utc", "in_zone", "flow_mw"]].rename(columns={"in_zone": "zone"})
    in_view = in_view.assign(
        exports_mw=(-in_view["flow_mw"]).clip(lower=0),
        imports_mw=in_view["flow_mw"].clip(lower=0),
    )

    combined = pd.concat(
        [
            out_view[["datetime_utc", "zone", "exports_mw", "imports_mw"]],
            in_view[["datetime_utc", "zone", "exports_mw", "imports_mw"]],
        ]
    )

    agg = combined.groupby(["datetime_utc", "zone"], as_index=False).sum()
    agg["net_export_mw"] = agg["exports_mw"] - agg["imports_mw"]
    return agg


def aggregate_zone_capacity(capacity_edges: pd.DataFrame) -> pd.DataFrame:
    if capacity_edges.empty:
        return pd.DataFrame(columns=["datetime_utc", "zone", "capacity_export_mw", "capacity_import_mw"])

    out_view = capacity_edges[["datetime_utc", "out_zone", "capacity_mw"]].rename(
        columns={"out_zone": "zone", "capacity_mw": "capacity_export_mw"}
    )
    in_view = capacity_edges[["datetime_utc", "in_zone", "capacity_mw"]].rename(
        columns={"in_zone": "zone", "capacity_mw": "capacity_import_mw"}
    )

    out_grouped = out_view.groupby(["datetime_utc", "zone"], as_index=False).sum()
    in_grouped = in_view.groupby(["datetime_utc", "zone"], as_index=False).sum()

    capacity = pd.merge(out_grouped, in_grouped, how="outer", on=["datetime_utc", "zone"])
    return capacity


def load_weather(weather_dir: Path, years: List[int]) -> pd.DataFrame:
    records = []
    for year in years:
        for month in range(1, 13):
            path = weather_dir / str(year) / f"bidding_zone_weather_{year}_{month:02d}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df["datetime_utc"] = pd.to_datetime(df["valid_time"], utc=True).dt.tz_convert(None)
            df["wind10"] = np.sqrt(np.square(df.get("u10", 0)) + np.square(df.get("v10", 0)))
            df["wind100"] = (
                np.sqrt(np.square(df["u100"]) + np.square(df["v100"]))
                if "u100" in df.columns and "v100" in df.columns
                else np.nan
            )
            for col in ["msl", "u10", "v10", "u100", "v100"]:
                if col not in df.columns:
                    df[col] = np.nan
            records.append(
                df[
                    [
                        "datetime_utc",
                        "zone",
                        "t2m",
                        "wind10",
                        "wind100",
                        "msl",
                        "u10",
                        "v10",
                        "u100",
                        "v100",
                    ]
                ].copy()
            )
    if not records:
        return pd.DataFrame(columns=["datetime_utc", "zone"])
    return pd.concat(records, ignore_index=True)


def load_esett_balancing_prices(esett_dir: Path, years: List[int]) -> pd.DataFrame:
    records = []
    for year in years:
        pattern = "balancing_price_*"
        path = next((esett_dir / str(year)).glob(pattern), None)
        if not path:
            continue
        df = pd.read_csv(path)
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True).dt.tz_convert(None)
        long = df.melt(id_vars="datetime_utc", var_name="zone_dir", value_name="price_eur_mwh")
        long[["zone", "direction"]] = long["zone_dir"].str.rsplit("_", n=1, expand=True)
        pivot = (
            long.pivot_table(
                index=["datetime_utc", "zone"],
                columns="direction",
                values="price_eur_mwh",
                aggfunc="first",
            )
            .reset_index()
            .rename_axis(None, axis=1)
            .rename(columns={"up": "bal_price_up_eur_mwh", "down": "bal_price_down_eur_mwh"})
        )
        records.append(pivot)
    if not records:
        return pd.DataFrame(columns=["datetime_utc", "zone", "bal_price_up_eur_mwh", "bal_price_down_eur_mwh"])
    return pd.concat(records, ignore_index=True)


def load_esett_volume(esett_dir: Path, years: List[int], kind: str, column_name: str) -> pd.DataFrame:
    records = []
    for year in years:
        path_csv = esett_dir / str(year) / f"{kind}_{year}.csv"
        path_alt = esett_dir / str(year) / f"{kind}_{year}_MWH.csv"
        path = path_csv if path_csv.exists() else path_alt if path_alt.exists() else None
        if path is None:
            continue
        df = pd.read_csv(path)
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True).dt.tz_convert(None)
        long = df.melt(id_vars="datetime_utc", var_name="zone", value_name=column_name)
        records.append(long)
    if not records:
        return pd.DataFrame(columns=["datetime_utc", "zone", column_name])
    return pd.concat(records, ignore_index=True)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df["hour_of_day"] = df["datetime_utc"].dt.hour
    df["weekday"] = df["datetime_utc"].dt.weekday
    df["month"] = df["datetime_utc"].dt.month
    df["year"] = df["datetime_utc"].dt.year
    season_map = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
    df["season"] = df["month"].map(season_map)
    return df


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(np.nan, index=series.index)
    return (series - series.mean()) / std


def compute_surplus_flags(panel: pd.DataFrame) -> pd.DataFrame:
    panel["net_surplus_mw"] = panel["gen_total_mw"] - panel["actual_load_mw"]
    panel["renewable_gen_mw"] = (
        panel["gen_wind_onshore_mw"]
        + panel["gen_wind_offshore_mw"]
        + panel["gen_hydro_ror_mw"]
        + panel["gen_hydro_reservoir_mw"]
        + panel["gen_solar_mw"]
    )
    panel["renewable_share"] = np.where(
        panel["gen_total_mw"].abs() > 0, panel["renewable_gen_mw"] / panel["gen_total_mw"], np.nan
    )

    panel["surplus_flag"] = (
        ((panel["net_surplus_mw"] > 0) & (panel["da_price_eur_mwh"] <= 5))
        | (panel["da_price_eur_mwh"] <= 0)
    ).astype("Int64")

    panel["z_net_surplus"] = panel.groupby("zone")["net_surplus_mw"].transform(zscore)
    panel["z_da_price"] = panel.groupby("zone")["da_price_eur_mwh"].transform(zscore)
    panel["z_net_export"] = panel.groupby("zone")["net_export_mw"].transform(zscore)
    panel["surplus_severity"] = panel["z_net_surplus"] - panel["z_da_price"] + panel["z_net_export"]
    return panel


def build_surplus_events(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()

    for col in ["wind100", "bal_price_down_eur_mwh", "abs_imbalance_mwh", "export_utilisation"]:
        if col not in panel.columns:
            panel[col] = np.nan

    panel = panel.sort_values(["zone", "datetime_utc"]).copy()
    prev_flag = panel.groupby("zone")["surplus_flag"].shift(fill_value=0)
    panel["event_start"] = (panel["surplus_flag"] == 1) & (prev_flag != 1)
    panel["surplus_event_id"] = panel.groupby("zone")["event_start"].cumsum()
    panel.loc[panel["surplus_flag"] != 1, "surplus_event_id"] = pd.NA

    events = (
        panel[panel["surplus_flag"] == 1]
        .groupby(["zone", "surplus_event_id"])
        .agg(
            start_time=("datetime_utc", "min"),
            end_time=("datetime_utc", "max"),
            duration_h=("datetime_utc", "count"),
            min_price=("da_price_eur_mwh", "min"),
            mean_price=("da_price_eur_mwh", "mean"),
            max_net_surplus_mw=("net_surplus_mw", "max"),
            mean_renewable_share=("renewable_share", "mean"),
            mean_wind100=("wind100", "mean"),
            mean_load_mw=("actual_load_mw", "mean"),
            mean_net_export_mw=("net_export_mw", "mean"),
            max_export_utilisation=("export_utilisation", "max"),
            mean_bal_price_down=("bal_price_down_eur_mwh", "mean"),
            mean_abs_imbalance=("abs_imbalance_mwh", "mean"),
            mean_severity=("surplus_severity", "mean"),
        )
        .reset_index()
    )
    return events


def build_pipeline(args: argparse.Namespace) -> None:
    base_entsoe = Path("Dataset") / "Entsoe"
    base_esett = Path("Dataset") / "esett"
    weather_dir = Path("Weather data")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ENTSO-E prices...")
    prices = load_entsoe_prices(base_entsoe, args.years, args.zones)

    print("Loading ENTSO-E load...")
    load_df = load_entsoe_load(base_entsoe, args.years, args.zones)

    print("Loading ENTSO-E generation...")
    generation = load_entsoe_generation(base_entsoe, args.years, args.zones)

    print("Loading cross-border flows and capacities...")
    flow_edges = load_flow_edges(base_entsoe, args.years, args.zones)
    capacity_edges = load_capacity_edges(base_entsoe, args.years, args.zones)

    flows = aggregate_zone_flows(flow_edges)
    capacity = aggregate_zone_capacity(capacity_edges)
    flows = pd.merge(flows, capacity, how="left", on=["datetime_utc", "zone"])
    for col in ["exports_mw", "imports_mw", "capacity_export_mw", "capacity_import_mw"]:
        if col in flows.columns:
            flows[col] = flows[col].fillna(0)
    flows["export_utilisation"] = flows["exports_mw"] / flows["capacity_export_mw"]
    flows["import_utilisation"] = flows["imports_mw"] / flows["capacity_import_mw"]
    flows.loc[flows["capacity_export_mw"] <= 0, "export_utilisation"] = np.nan
    flows.loc[flows["capacity_import_mw"] <= 0, "import_utilisation"] = np.nan
    flows = flows[flows["zone"].isin(args.zones)]

    print("Loading weather...")
    weather = load_weather(weather_dir, args.years)

    print("Loading balancing prices...")
    bal_prices = load_esett_balancing_prices(base_esett, args.years)

    print("Loading eSett production/consumption/imbalance...")
    esett_prod = load_esett_volume(base_esett, args.years, "production", "esett_production_mwh")
    esett_cons = load_esett_volume(base_esett, args.years, "consumption", "esett_consumption_mwh")
    esett_imb = load_esett_volume(base_esett, args.years, "imbalance", "imbalance_mwh")
    if not esett_imb.empty:
        esett_imb["abs_imbalance_mwh"] = esett_imb["imbalance_mwh"].abs()

    if prices.empty:
        print("No price data found; aborting.")
        return

    panel = prices[["datetime_utc", "zone", "da_price_eur_mwh"]].copy()
    for df in [load_df, generation, flows, weather, bal_prices, esett_prod, esett_cons, esett_imb]:
        if df is None or df.empty:
            continue
        panel = pd.merge(panel, df, how="left", on=["datetime_utc", "zone"])

    panel = panel.dropna(subset=["datetime_utc"])
    panel = panel.sort_values(["zone", "datetime_utc"]).reset_index(drop=True)
    panel = panel.dropna(
        subset=["da_price_eur_mwh", "actual_load_mw", "gen_total_mw", "wind10", "wind100", "t2m"],
        how="any",
    )
    # fill remaining numeric gaps (e.g., balancing/imbalance) with zero
    num_cols = panel.select_dtypes(include=["number"]).columns
    panel[num_cols] = panel[num_cols].fillna(0)
    panel = add_calendar_features(panel)
    panel = compute_surplus_flags(panel)

    print("Building surplus events...")
    events = build_surplus_events(panel)

    master_path = output_dir / "master_panel.parquet"
    events_path = output_dir / "surplus_events.parquet"

    panel.to_parquet(master_path, index=False)
    events.to_parquet(events_path, index=False)

    print(f"Saved hourly panel to {master_path}")
    print(f"Saved surplus events to {events_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build surplus analysis datasets.")
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS, help="Years to include")
    parser.add_argument("--zones", nargs="+", type=str, default=DEFAULT_ZONES, help="Zone codes to include")
    parser.add_argument("--output-dir", type=str, default="processed", help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    build_pipeline(parse_args())
