"""
Unemployment Analysis — India (2019–2020)
Batch script: data cleaning, exploration, visualization, COVID-19 impact, policy insights.
"""

from src.analytics import detect_anomalies, period_averages, state_lockdown_impact, top_bottom_states
from src.config import OUTPUT_DIR, UNEMP_COL
from src.data_loader import load_area_data, load_state_data

# Re-export for backward compatibility
load_and_clean_state_data = load_state_data
load_and_clean_area_data = load_area_data

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

OUTPUT_DIR.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120


def print_exploration(state_df, area_df):
    print("=" * 70)
    print("DATA EXPLORATION SUMMARY")
    print("=" * 70)
    print(f"\nState records: {len(state_df):,} | Regions: {state_df['Region'].nunique()}")
    print(f"Date range: {state_df['Date'].min().date()} to {state_df['Date'].max().date()}")
    print(state_df[UNEMP_COL].describe().round(2))
    print(f"\nRural/Urban records: {len(area_df):,}")
    print(area_df.groupby("Area")[UNEMP_COL].mean().round(2))


def plot_national_trend(state_df):
    monthly = state_df.groupby("Date")[UNEMP_COL].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(monthly["Date"], monthly["mean"], marker="o", linewidth=2)
    ax.fill_between(monthly["Date"], monthly["mean"] - monthly["std"],
                    monthly["mean"] + monthly["std"], alpha=0.2)
    ax.axvspan("2020-03-25", "2020-05-31", color="#e74c3c", alpha=0.15)
    ax.set_title("National Unemployment Trend")
    ax.set_ylabel("Unemployment Rate (%)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_national_trend.png")
    plt.close(fig)


def plot_covid_impact(state_df, area_df):
    periods = period_averages(state_df)
    impact = state_lockdown_impact(state_df)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(periods.index, periods.values, color=["#2ecc71", "#e74c3c", "#3498db"])
    axes[0].set_title("COVID Period Comparison")
    axes[0].tick_params(axis="x", rotation=15)

    top = impact.nlargest(10, "unemp_change_pp")
    axes[1].barh(top["Region"], top["unemp_change_pp"], color="#e67e22")
    axes[1].set_title("Top 10 Lockdown Increases")
    axes[1].invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_covid_impact.png")
    plt.close(fig)
    return impact.set_index("Region")["unemp_change_pp"]


def plot_seasonal_patterns(area_df):
    baseline = area_df[area_df["Date"] < "2020-01-01"].copy()
    baseline["Month"] = baseline["Date"].dt.month
    monthly = baseline.groupby(["Month", "Area"])[UNEMP_COL].mean().unstack()
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in monthly.columns:
        ax.plot(monthly.index, monthly[col], marker="o", label=col)
    ax.set_title("Seasonal Pattern (2019)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_seasonal_patterns.png")
    plt.close(fig)


def plot_regional_heatmap(state_df):
    pivot = state_df.pivot_table(index="Region", columns=state_df["Date"].dt.to_period("M"),
                                  values=UNEMP_COL)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_state_heatmap.png")
    plt.close(fig)


def plot_zone_comparison(state_df):
    zone = state_df.groupby(["Date", "Zone"])[UNEMP_COL].mean().reset_index()
    fig, ax = plt.subplots(figsize=(11, 5))
    for z in zone["Zone"].unique():
        s = zone[zone["Zone"] == z]
        ax.plot(s["Date"], s[UNEMP_COL], marker="o", label=z)
    ax.axvspan("2020-03-25", "2020-05-31", color="#e74c3c", alpha=0.1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_zone_trends.png")
    plt.close(fig)


def plot_employment_vs_unemployment(state_df):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(state_df["Estimated Employed"] / 1e6, state_df[UNEMP_COL],
               c=state_df["Date"].map(lambda d: d.toordinal()), cmap="viridis", alpha=0.7)
    ax.set_xlabel("Employed (millions)")
    ax.set_ylabel("Unemployment (%)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "08_employment_scatter.png")
    plt.close(fig)


def print_policy_insights(state_df, area_df, state_changes):
    periods = period_averages(state_df)
    rankings = top_bottom_states(state_df)
    anomalies = detect_anomalies(state_df)
    print("\n" + "=" * 70)
    print("POLICY INSIGHTS")
    print("=" * 70)
    print(f"Pre-COVID: {periods['Pre-COVID']:.1f}% | Lockdown: {periods['Lockdown']:.1f}% | Recovery: {periods['Recovery']:.1f}%")
    print(f"Worst states: {', '.join(rankings['worst_lockdown']['Region'].head(3))}")
    print(f"Anomalies detected: {len(anomalies)}")
    print(f"Charts saved to: {OUTPUT_DIR.resolve()}")


def main():
    state_df = load_state_data()
    area_df = load_area_data()
    print_exploration(state_df, area_df)
    plot_national_trend(state_df)
    changes = plot_covid_impact(state_df, area_df)
    plot_seasonal_patterns(area_df)
    plot_regional_heatmap(state_df)
    plot_zone_comparison(state_df)
    plot_employment_vs_unemployment(state_df)
    print_policy_insights(state_df, area_df, changes)


if __name__ == "__main__":
    main()
