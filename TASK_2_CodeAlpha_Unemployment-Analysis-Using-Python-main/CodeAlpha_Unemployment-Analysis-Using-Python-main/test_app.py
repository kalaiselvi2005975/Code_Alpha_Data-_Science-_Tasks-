"""Smoke test — run all dashboard code paths without Streamlit UI."""

from src.analytics import (
    correlation_matrix,
    detect_anomalies,
    period_averages,
    rolling_trend,
    seasonal_decomposition_table,
    simple_forecast,
    state_lockdown_impact,
    statistical_tests,
    top_bottom_states,
)
from src.charts import (
    anomaly_chart,
    correlation_heatmap,
    forecast_chart,
    geo_map_chart,
    national_trend_chart,
    period_comparison_chart,
    rural_urban_period_chart,
    seasonal_chart,
    state_heatmap,
    state_impact_chart,
    zone_trend_chart,
)
from src.config import UNEMP_COL
from src.data_loader import load_area_data, load_state_data


def run_all():
    state_df = load_state_data()
    area_df = load_area_data()

    # Normal data
    periods = period_averages(state_df)
    monthly = state_df.groupby("Date")[UNEMP_COL].agg(mean="mean", std="std").reset_index()
    rolling = rolling_trend(state_df)
    monthly = monthly.merge(rolling[["Date", "Rolling Mean"]], on="Date", how="left")

    national_trend_chart(monthly)
    geo_map_chart(state_df, state_df["Date"].iloc[5])
    state_heatmap(state_df.pivot_table(index="Region", columns=state_df["Date"].dt.to_period("M").astype(str), values=UNEMP_COL))
    zone_trend_chart(state_df.groupby(["Date", "Zone"])[UNEMP_COL].mean().reset_index())
    period_comparison_chart(periods.to_dict())
    state_impact_chart(state_lockdown_impact(state_df))
    correlation_heatmap(correlation_matrix(state_df))
    anomaly_chart(detect_anomalies(state_df, 2.0))
    anomaly_chart(detect_anomalies(state_df, 99.0))  # empty case
    forecast_chart(rolling[["Date", UNEMP_COL]], simple_forecast(rolling, 3))
    seasonal = seasonal_decomposition_table(area_df)
    seasonal_chart(seasonal)
    area_periods = period_averages(area_df, group_col="Area").pivot(index="Area", columns="Period", values=UNEMP_COL).reset_index()
    rural_urban_period_chart(area_periods)
    statistical_tests(state_df)
    top_bottom_states(state_df)

    # Edge cases — empty filters
    empty = state_df.iloc[0:0]
    geo_map_chart(empty, state_df["Date"].iloc[0])
    detect_anomalies(empty)
    state_lockdown_impact(empty)
    state_impact_chart(empty)
    statistical_tests(empty)

    print("ALL TESTS PASSED - No errors")


if __name__ == "__main__":
    run_all()
