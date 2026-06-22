"""Plotly chart builders for the Streamlit dashboard."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import (
    COLOR_LOCK,
    COLOR_PRE,
    COLOR_RECOVERY,
    EMP_COL,
    LOCKDOWN_END,
    LOCKDOWN_START,
    LPR_COL,
    MONTH_NAMES,
    UNEMP_COL,
)


def national_trend_chart(monthly: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["Date"], y=monthly["mean"],
        mode="lines+markers", name="Mean",
        line=dict(color="#3498db", width=2.5),
    ))
    if "Rolling Mean" in monthly.columns:
        fig.add_trace(go.Scatter(
            x=monthly["Date"], y=monthly["Rolling Mean"],
            mode="lines", name="3-Month Rolling Avg",
            line=dict(color="#9b59b6", width=2, dash="dash"),
        ))
    if "std" in monthly.columns:
        fig.add_trace(go.Scatter(
            x=monthly["Date"], y=monthly["mean"] + monthly["std"],
            mode="lines", line=dict(width=0), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=monthly["Date"], y=monthly["mean"] - monthly["std"],
            mode="lines", fill="tonexty", fillcolor="rgba(52,152,219,0.12)",
            line=dict(width=0), name="±1 Std Dev",
        ))
    fig.add_vrect(
        x0=LOCKDOWN_START, x1=LOCKDOWN_END,
        fillcolor="rgba(231,76,60,0.12)", line_width=0,
        annotation_text="Lockdown", annotation_position="top left",
    )
    fig.update_layout(
        title="National Unemployment Trend",
        xaxis_title="Month", yaxis_title="Unemployment Rate (%)",
        height=440, hovermode="x unified", template="plotly_white",
    )
    return fig


def geo_map_chart(state_df: pd.DataFrame, date: pd.Timestamp) -> go.Figure:
    """Bubble map of unemployment by state for a selected month."""
    if state_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available for selected filters", showarrow=False)
        fig.update_layout(title="Geographic Unemployment Map", height=480)
        return fig

    snapshot = state_df[state_df["Date"] == date].dropna(subset=["latitude", "longitude"])
    if snapshot.empty:
        nearest = state_df["Date"].unique()
        if len(nearest) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No geographic data available", showarrow=False)
            fig.update_layout(title="Geographic Unemployment Map", height=480)
            return fig
        date = min(nearest, key=lambda d: abs(d - date))
        snapshot = state_df[state_df["Date"] == date].dropna(subset=["latitude", "longitude"])

    if snapshot.empty:
        fig = go.Figure()
        fig.add_annotation(text="No latitude/longitude data for this period", showarrow=False)
        fig.update_layout(title="Geographic Unemployment Map", height=480)
        return fig

    fig = px.scatter_geo(
        snapshot,
        lat="latitude",
        lon="longitude",
        size=UNEMP_COL,
        color=UNEMP_COL,
        hover_name="Region",
        hover_data={"Zone": True, EMP_COL: ":,.0f", LPR_COL: ":.1f"},
        color_continuous_scale="YlOrRd",
        size_max=35,
        title=f"Geographic Unemployment Map — {date.strftime('%b %Y')}",
    )
    fig.update_geos(
        scope="asia",
        center=dict(lat=22.5, lon=79),
        projection_scale=4.5,
        showland=True, landcolor="#f0f0f0",
        showcountries=True, countrycolor="#ccc",
    )
    fig.update_layout(height=480, margin=dict(l=0, r=0, t=40, b=0))
    return fig


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    labels = ["Unemployment", "Employment", "Labour Participation"]
    fig = px.imshow(
        corr.values,
        x=labels, y=labels,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=".2f",
        title="Correlation Matrix — Labour Market Indicators",
    )
    fig.update_layout(height=380)
    return fig


def period_comparison_chart(periods: dict) -> go.Figure:
    colors = [COLOR_PRE, COLOR_LOCK, COLOR_RECOVERY]
    fig = go.Figure(go.Bar(
        x=list(periods.keys()), y=list(periods.values()),
        marker_color=colors,
        text=[f"{v:.1f}%" for v in periods.values()],
        textposition="outside",
    ))
    fig.update_layout(
        title="Unemployment by COVID Period",
        yaxis_title="Avg Rate (%)", height=380,
        template="plotly_white", showlegend=False,
    )
    return fig


def state_impact_chart(impact_df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    if impact_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No impact data available", showarrow=False)
        fig.update_layout(height=420, template="plotly_white")
        return fig

    top = impact_df.nlargest(top_n, "unemp_change_pp")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Lockdown Increase (pp)", "Recovery Gap (pp)"))
    fig.add_trace(
        go.Bar(x=top["unemp_change_pp"], y=top["Region"], orientation="h", marker_color="#e67e22", name="Increase"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=top["recovery_gap_pp"], y=top["Region"], orientation="h", marker_color="#3498db", name="Recovery Gap"),
        row=1, col=2,
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=420, showlegend=False, template="plotly_white")
    return fig


def anomaly_chart(anomalies: pd.DataFrame) -> go.Figure:
    if anomalies.empty:
        fig = go.Figure()
        fig.add_annotation(text="No anomalies detected at current threshold", showarrow=False)
        return fig
    fig = px.scatter(
        anomalies, x="Date", y=UNEMP_COL,
        color="Type", symbol="Region",
        size=anomalies["Z-Score"].abs(),
        hover_data=["Region", "Z-Score", "Zone"],
        title="Anomaly Detection — Unemployment Spikes & Dips",
        color_discrete_map={"Spike": "#e74c3c", "Dip": "#2ecc71"},
    )
    fig.update_layout(height=400, template="plotly_white")
    return fig


def forecast_chart(monthly: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["Date"], y=monthly[UNEMP_COL],
        mode="lines+markers", name="Historical",
        line=dict(color="#3498db"),
    ))
    if not forecast.empty:
        fig.add_trace(go.Scatter(
            x=forecast["Date"], y=forecast[UNEMP_COL],
            mode="lines+markers", name="Linear Forecast",
            line=dict(color="#e67e22", dash="dash"),
        ))
    fig.update_layout(
        title="Unemployment Trend & Simple Forecast",
        xaxis_title="Month", yaxis_title="Rate (%)",
        height=400, template="plotly_white",
    )
    return fig


def seasonal_chart(seasonal_df: pd.DataFrame) -> go.Figure:
    seasonal_df = seasonal_df.copy()
    seasonal_df["MonthName"] = seasonal_df["Month"].map(lambda m: MONTH_NAMES[m - 1])
    fig = px.line(
        seasonal_df, x="MonthName", y="avg_unemployment",
        color="Area", markers=True,
        category_orders={"MonthName": MONTH_NAMES},
        labels={"avg_unemployment": "Unemployment Rate (%)", "MonthName": "Month"},
        title="Seasonal Pattern — 2019 Baseline",
    )
    fig.update_layout(height=400, template="plotly_white")
    return fig


def rural_urban_period_chart(area_periods: pd.DataFrame) -> go.Figure:
    melted = area_periods.melt(id_vars="Area", var_name="Period", value_name="Rate")
    fig = px.bar(
        melted, x="Area", y="Rate", color="Period", barmode="group",
        color_discrete_map={"Pre-COVID": COLOR_PRE, "Lockdown": COLOR_LOCK, "Recovery": COLOR_RECOVERY},
        labels={"Rate": "Unemployment Rate (%)"},
        title="Rural vs Urban — COVID Period Comparison",
    )
    fig.update_layout(height=400, template="plotly_white")
    return fig


def zone_trend_chart(zone_monthly: pd.DataFrame) -> go.Figure:
    fig = px.line(
        zone_monthly, x="Date", y=UNEMP_COL,
        color="Zone", markers=True,
        title="Unemployment by Geographic Zone",
    )
    fig.add_vrect(
        x0=LOCKDOWN_START, x1=LOCKDOWN_END,
        fillcolor="rgba(231,76,60,0.1)", line_width=0,
    )
    fig.update_layout(height=400, template="plotly_white")
    return fig


def state_heatmap(pivot: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        color_continuous_scale="YlOrRd",
        aspect="auto",
        labels=dict(color="Unemployment %"),
        title="State × Month Heatmap",
    )
    fig.update_layout(height=520)
    return fig
