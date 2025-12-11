from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


OUTPUT_DIR = Path("analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
MASTER_PATH = Path("processed") / "master_panel.parquet"
EVENTS_PATH = Path("processed") / "surplus_events.parquet"

sns.set_theme(style="whitegrid", context="talk", palette="tab10")
plt.rcParams.update({"axes.spines.right": True, "axes.spines.top": False})


def load_panel() -> pd.DataFrame:
    panel = pd.read_parquet(MASTER_PATH)
    panel["datetime_utc"] = pd.to_datetime(panel["datetime_utc"]).dt.tz_localize(None)
    panel["date"] = panel["datetime_utc"].dt.date
    panel["surplus_flag"] = panel["surplus_flag"].astype(int)
    return panel


def scatter_price_surplus(panel: pd.DataFrame) -> None:
    sample = panel.sample(n=120000, random_state=42) if len(panel) > 120000 else panel
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=sample,
        x="net_surplus_mw",
        y="da_price_eur_mwh",
        hue="zone",
        alpha=0.35,
        s=12,
        linewidth=0,
    )
    plt.axhline(0, color="grey", linestyle="--", linewidth=1, label="Price = 0")
    plt.axvline(0, color="grey", linestyle=":", linewidth=1, label="Net balance = 0")
    plt.title("Price vs net surplus (sampled hours)")
    plt.xlabel("Net surplus (MW)")
    plt.ylabel("Day-ahead price (EUR/MWh)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "price_vs_surplus_scatter.png", dpi=250, bbox_inches="tight")
    plt.close()


def calendar_heatmap(panel: pd.DataFrame, zone: str, year: int) -> None:
    sub = panel[(panel["zone"] == zone) & (panel["datetime_utc"].dt.year == year)]
    if sub.empty:
        return
    daily = sub.groupby("date")["surplus_flag"].sum().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["month"] = daily["date"].dt.month
    daily["day"] = daily["date"].dt.day
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = daily.pivot(index="month", columns="day", values="surplus_flag")
    plt.figure(figsize=(14, 4.5))
    sns.heatmap(
        pivot,
        cmap="YlGnBu",
        cbar_kws={"label": "Surplus hours per day"},
        linewidths=0.2,
        linecolor="white",
    )
    plt.title(f"{zone} {year}: Surplus hours per day")
    plt.xlabel("Day of month")
    plt.ylabel("Month")
    plt.yticks(ticks=[i + 0.5 for i in range(12)], labels=month_labels[: pivot.index.max()], rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"calendar_heatmap_{zone}_{year}.png", dpi=250, bbox_inches="tight")
    plt.close()


def calendar_heatmap_multizone(panel: pd.DataFrame, year: int) -> None:
    """Multi-zone 'calendar' heatmap: rows = zones, columns = day-of-year, values = surplus hours per day."""
    sub = panel[panel["datetime_utc"].dt.year == year]
    if sub.empty:
        return
    sub = sub.copy()
    sub["doy"] = sub["datetime_utc"].dt.dayofyear
    daily = sub.groupby(["zone", "doy"])["surplus_flag"].sum().reset_index()
    pivot = daily.pivot(index="zone", columns="doy", values="surplus_flag").fillna(0)

    plt.figure(figsize=(16, 6))
    sns.heatmap(
        pivot,
        cmap="YlGnBu",
        cbar_kws={"label": "Surplus hours per day"},
        linewidths=0.0,
    )
    plt.title(f"Multi-zone surplus calendar – {year}")
    plt.xlabel("Day of year")
    plt.ylabel("Zone")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"calendar_heatmap_multizone_{year}.png", dpi=250, bbox_inches="tight")
    plt.close()


def price_path_top_event(panel: pd.DataFrame, events: pd.DataFrame, zone: str, window: int = 12) -> None:
    zone_events = events[events["zone"] == zone]
    if zone_events.empty:
        return
    top = zone_events.sort_values("mean_severity", ascending=False).iloc[0]
    start = pd.to_datetime(top["start_time"]).tz_localize(None)
    end = pd.to_datetime(top["end_time"]).tz_localize(None)
    sub = panel[(panel["zone"] == zone)].copy()
    sub["tau"] = (sub["datetime_utc"] - start).dt.total_seconds() / 3600
    sub = sub[(sub["tau"] >= -window) & (sub["tau"] <= window)]
    if sub.empty:
        return
    sub = sub.groupby("tau").agg(
        price=("da_price_eur_mwh", "median"),
        net_surplus=("net_surplus_mw", "median"),
    ).reset_index()
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(sub["tau"], sub["price"], color="tab:blue", label="Price (EUR/MWh)", linewidth=2)
    ax1.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax1.axvline(0, color="black", linestyle="--", linewidth=1, label="Event start")
    ax1.set_xlabel("Hours since event start")
    ax1.set_ylabel("Day-ahead price (EUR/MWh)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(sub["tau"], sub["net_surplus"], color="tab:green", label="Net surplus (MW)", linewidth=2)
    ax2.set_ylabel("Net surplus (MW)", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    fig.suptitle(f"{zone}: price & net surplus around top event\n({start} to {end})", y=1.02)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=9, frameon=True)
    fig.tight_layout()
    plt.savefig(OUTPUT_DIR / f"price_path_top_event_{zone}.png", dpi=250, bbox_inches="tight")
    plt.close()


def main() -> None:
    panel = load_panel()
    events = pd.read_parquet(EVENTS_PATH)

    scatter_price_surplus(panel)

    zones = sorted(panel["zone"].unique())
    years = sorted(panel["datetime_utc"].dt.year.unique())
    for zone in zones:
        for year in years:
            calendar_heatmap(panel, zone=zone, year=year)
        price_path_top_event(panel, events, zone=zone, window=12)

    for year in years:
        calendar_heatmap_multizone(panel, year=year)

    print(f"Plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
