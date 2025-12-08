"""
Lightweight analysis on the processed master panel to cover the project steps:
- surplus vs non-surplus contrasts
- correlations with key drivers
- logistic regression for surplus probability
- ranked surplus events

Outputs are written to analysis_outputs/.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path("analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MASTER_PATH = Path("processed") / "master_panel.parquet"
EVENTS_PATH = Path("processed") / "surplus_events.parquet"


def load_master() -> pd.DataFrame:
    panel = pd.read_parquet(MASTER_PATH)
    # ensure correct dtypes
    panel["surplus_flag"] = panel["surplus_flag"].astype("Int64")
    return panel


def surplus_vs_non(panel: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "hours": ("surplus_flag", "count"),
        "mean_price": ("da_price_eur_mwh", "mean"),
        "mean_net_surplus_mw": ("net_surplus_mw", "mean"),
        "mean_net_export_mw": ("net_export_mw", "mean"),
        "mean_export_util": ("export_utilisation", "mean"),
        "mean_renewable_share": ("renewable_share", "mean"),
        "mean_wind100": ("wind100", "mean"),
        "mean_load_mw": ("actual_load_mw", "mean"),
        "mean_bal_price_down": ("bal_price_down_eur_mwh", "mean"),
        "mean_abs_imbalance": ("abs_imbalance_mwh", "mean"),
    }
    grouped = (
        panel.groupby(["zone", "surplus_flag"])
        .agg(**metrics)
        .reset_index()
        .rename(columns={"surplus_flag": "is_surplus"})
    )
    grouped.to_csv(OUTPUT_DIR / "surplus_vs_non.csv", index=False)
    return grouped


def correlation_table(panel: pd.DataFrame) -> pd.DataFrame:
    drivers = ["renewable_share", "wind100", "t2m", "actual_load_mw", "net_export_mw", "export_utilisation"]
    records: List[Dict] = []
    for zone, df in panel.groupby("zone"):
        for driver in drivers:
            sub = df[["net_surplus_mw", driver]].dropna()
            if len(sub) < 50:
                continue
            r, p = stats.pearsonr(sub["net_surplus_mw"], sub[driver])
            records.append({"zone": zone, "driver": driver, "pearson_r": r, "p_value": p})
    corr_df = pd.DataFrame(records)
    corr_df.to_csv(OUTPUT_DIR / "correlations_net_surplus.csv", index=False)
    return corr_df


def prepare_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = df.dropna(subset=feature_cols + ["surplus_flag"])
    if df.empty:
        return np.empty((0, len(feature_cols))), np.array([]), feature_cols
    X = df[feature_cols].values
    y = df["surplus_flag"].astype(int).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, feature_cols


def logistic_per_zone(panel: pd.DataFrame) -> pd.DataFrame:
    feature_cols = ["actual_load_mw", "renewable_share", "wind100", "t2m", "net_export_mw"]
    records: List[Dict] = []
    for zone, df in panel.groupby("zone"):
        X, y, cols = prepare_features(df.copy(), feature_cols)
        if len(y) < 200 or len(np.unique(y)) < 2:
            continue
        clf = LogisticRegression(max_iter=200, solver="lbfgs")
        clf.fit(X, y)
        odds = np.exp(clf.coef_[0])
        for col, od in zip(cols, odds):
            records.append({"zone": zone, "feature": col, "odds_ratio": od})
    res = pd.DataFrame(records)
    res.to_csv(OUTPUT_DIR / "logit_surplus_odds.csv", index=False)
    return res


def rank_events() -> pd.DataFrame:
    if not EVENTS_PATH.exists():
        return pd.DataFrame()
    events = pd.read_parquet(EVENTS_PATH)
    events_sorted = events.sort_values("mean_severity", ascending=False)
    top20 = events_sorted.groupby("zone").head(20)
    top20.to_csv(OUTPUT_DIR / "top20_events_by_zone.csv", index=False)
    return top20


def main() -> None:
    panel = load_master()
    print("Surplus vs non-surplus summary...")
    surplus_vs_non(panel)

    print("Correlation table...")
    correlation_table(panel)

    print("Logistic regression odds ratios...")
    logistic_per_zone(panel)

    print("Ranking events...")
    rank_events()

    print(f"Outputs written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
