"""
Advanced Streamlit Dashboard — Unemployment Analysis (India 2019–2020)
Features: geo maps, anomaly detection, correlation analysis, forecasting, exports.
"""

import io

import pandas as pd
import plotly.express as px
import streamlit as st

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
from src.config import EMP_COL, LPR_COL, UNEMP_COL
from src.data_loader import get_data_summary, load_area_data, load_state_data

# ── Page config ──
st.set_page_config(
    page_title="Unemployment Analytics Pro | India",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0; }
    .sub-header { color: #666; font-size: 1rem; margin-bottom: 1.5rem; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea11, #764ba211);
        border: 1px solid #667eea33;
        border-radius: 10px; padding: 12px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading & cleaning datasets…")
def load_data():
    return load_state_data(), load_area_data()


def export_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def main():
    state_df, area_df = load_data()
    summary = get_data_summary(state_df, area_df)

    # ── Sidebar ──
    st.sidebar.image("https://img.icons8.com/fluency/96/combo-chart.png", width=64)
    st.sidebar.title("Control Panel")

    page = st.sidebar.radio(
        "Navigate",
        ["🏠 Overview", "📈 Trends & Maps", "🦠 COVID Analysis", "🔬 Advanced Analytics",
         "🌾 Rural vs Urban", "📅 Seasonal", "💡 Policy & Export"],
    )

    st.sidebar.divider()
    st.sidebar.subheader("Filters")
    selected_states = st.sidebar.multiselect(
        "State/UT", sorted(state_df["Region"].unique()),
        default=sorted(state_df["Region"].unique()),
    )
    selected_zones = st.sidebar.multiselect(
        "Zone", summary["zones"], default=summary["zones"],
    )
    area_filter = st.sidebar.multiselect("Area Type", ["Rural", "Urban"], default=["Rural", "Urban"])
    anomaly_threshold = st.sidebar.slider("Anomaly Z-Score Threshold", 1.5, 3.5, 2.0, 0.1)

    filtered_state = state_df[
        state_df["Region"].isin(selected_states) & state_df["Zone"].isin(selected_zones)
    ]
    filtered_area = area_df[
        area_df["Region"].isin(selected_states) & area_df["Area"].isin(area_filter)
    ]

    if filtered_state.empty:
        st.warning("No data matches your filters. Please select at least one State/UT and Zone.")
        return

    # ── Header ──
    st.markdown('<p class="main-header">📊 Unemployment Analytics Pro — India</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Advanced labour market intelligence · COVID-19 impact · Seasonal patterns · Policy insights</p>',
        unsafe_allow_html=True,
    )

    # ── KPIs ──
    periods = period_averages(filtered_state)
    pre, lock, recovery = periods["Pre-COVID"], periods["Lockdown"], periods["Recovery"]
    avg_lpr = filtered_state[LPR_COL].mean()
    stats_result = statistical_tests(filtered_state)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Pre-COVID", f"{pre:.1f}%" if pd.notna(pre) else "N/A")
    k2.metric("Lockdown Peak", f"{lock:.1f}%" if pd.notna(lock) else "N/A",
              delta=f"{lock - pre:+.1f} pp" if pd.notna(lock) and pd.notna(pre) else None,
              delta_color="inverse")
    k3.metric("Recovery", f"{recovery:.1f}%" if pd.notna(recovery) else "N/A",
              delta=f"{recovery - pre:+.1f} pp" if pd.notna(recovery) and pd.notna(pre) else None)
    k4.metric("Labour Participation", f"{avg_lpr:.1f}%" if pd.notna(avg_lpr) else "N/A")
    p_val = stats_result.get("p_value", float("nan"))
    k5.metric("COVID Significance", "Yes" if stats_result["significant_95"] else "No",
              help=f"p-value = {p_val:.4f} (t-test pre vs lockdown)" if pd.notna(p_val) else "Insufficient data")

    st.divider()

    # ══════════════════════════════════════════
    # PAGE: Overview
    # ══════════════════════════════════════════
    if page == "🏠 Overview":
        c1, c2 = st.columns([2, 1])
        with c1:
            monthly = (
                filtered_state.groupby("Date")[UNEMP_COL]
                .agg(mean="mean", std="std").reset_index()
            )
            rolling = rolling_trend(filtered_state)
            monthly = monthly.merge(rolling[["Date", "Rolling Mean"]], on="Date", how="left")
            st.plotly_chart(national_trend_chart(monthly), use_container_width=True)

        with c2:
            st.subheader("Dataset Summary")
            st.markdown(f"""
| Metric | Value |
|--------|-------|
| State records | {summary['state_records']:,} |
| Rural/Urban records | {summary['area_records']:,} |
| States/UTs | {summary['states']} |
| State data range | {summary['state_date_min'].strftime('%b %Y')} – {summary['state_date_max'].strftime('%b %Y')} |
| Area data range | {summary['area_date_min'].strftime('%b %Y')} – {summary['area_date_max'].strftime('%b %Y')} |
            """)
            st.subheader("Quick Stats")
            st.write(filtered_state[UNEMP_COL].describe().round(2))

        rankings = top_bottom_states(filtered_state)
        r1, r2, r3 = st.columns(3)
        with r1:
            st.subheader("🔺 Largest Lockdown Spikes")
            st.dataframe(rankings["worst_lockdown"], hide_index=True, use_container_width=True)
        with r2:
            st.subheader("✅ Best Recovery")
            st.dataframe(rankings["best_recovery"], hide_index=True, use_container_width=True)
        with r3:
            st.subheader("🔴 Highest Lockdown Rate")
            st.dataframe(rankings["highest_unemployment"], hide_index=True, use_container_width=True)

    # ══════════════════════════════════════════
    # PAGE: Trends & Maps
    # ══════════════════════════════════════════
    elif page == "📈 Trends & Maps":
        tab_trend, tab_map, tab_heat = st.tabs(["Trends", "Geo Map", "Heatmap"])

        with tab_trend:
            monthly = filtered_state.groupby("Date")[UNEMP_COL].agg(mean="mean", std="std").reset_index()
            rolling = rolling_trend(filtered_state)
            monthly = monthly.merge(rolling[["Date", "Rolling Mean"]], on="Date", how="left")
            st.plotly_chart(national_trend_chart(monthly), use_container_width=True)

            compare_states = st.multiselect(
                "Compare States", sorted(filtered_state["Region"].unique()),
                default=sorted(filtered_state["Region"].unique())[:3],
            )
            if compare_states:
                comp = filtered_state[filtered_state["Region"].isin(compare_states)]
                fig = px.line(
                    comp, x="Date", y=UNEMP_COL, color="Region", markers=True,
                    title="State-Level Comparison",
                )
                fig.add_vrect(x0="2020-03-25", x1="2020-05-31", fillcolor="rgba(231,76,60,0.1)", line_width=0)
                st.plotly_chart(fig, use_container_width=True)

            zone_monthly = filtered_state.groupby(["Date", "Zone"])[UNEMP_COL].mean().reset_index()
            st.plotly_chart(zone_trend_chart(zone_monthly), use_container_width=True)

        with tab_map:
            available_dates = sorted(filtered_state["Date"].unique())
            if not available_dates:
                st.info("No dates available for the geo map with current filters.")
            else:
                selected_date = st.select_slider(
                    "Select Month",
                    options=available_dates,
                    value=available_dates[len(available_dates) // 2],
                    format_func=lambda d: d.strftime("%b %Y"),
                )
                st.plotly_chart(geo_map_chart(filtered_state, selected_date), use_container_width=True)

        with tab_heat:
            pivot = filtered_state.pivot_table(
                index="Region",
                columns=filtered_state["Date"].dt.to_period("M").astype(str),
                values=UNEMP_COL,
            )
            st.plotly_chart(state_heatmap(pivot), use_container_width=True)

    # ══════════════════════════════════════════
    # PAGE: COVID Analysis
    # ══════════════════════════════════════════
    elif page == "🦠 COVID Analysis":
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(period_comparison_chart(periods.to_dict()), use_container_width=True)
        with col2:
            impact = state_lockdown_impact(filtered_state)
            st.plotly_chart(state_impact_chart(impact), use_container_width=True)

        st.subheader("State Impact Table")
        st.dataframe(
            impact.style.background_gradient(subset=["unemp_change_pp", "recovery_gap_pp"], cmap="YlOrRd"),
            use_container_width=True, height=400,
        )

        fig = px.scatter(
            filtered_state,
            x=filtered_state[EMP_COL] / 1e6,
            y=UNEMP_COL,
            color="Date",
            hover_data=["Region", "Zone", LPR_COL],
            labels={"x": "Employed (millions)", "y": "Unemployment (%)"},
            title="Employment vs Unemployment",
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════
    # PAGE: Advanced Analytics
    # ══════════════════════════════════════════
    elif page == "🔬 Advanced Analytics":
        tab_corr, tab_anomaly, tab_forecast = st.tabs(["Correlation", "Anomalies", "Forecast"])

        with tab_corr:
            corr = correlation_matrix(filtered_state)
            st.plotly_chart(correlation_heatmap(corr), use_container_width=True)
            st.markdown("""
**Interpretation:**
- Strong negative correlation between unemployment and employment confirms labour market contraction during shocks.
- Labour participation often moves independently — workers may exit the labour force without being counted as unemployed.
            """)

        with tab_anomaly:
            anomalies = detect_anomalies(filtered_state, z_threshold=anomaly_threshold)
            st.plotly_chart(anomaly_chart(anomalies), use_container_width=True)
            if anomalies.empty:
                st.info("No anomalies found. Try lowering the Z-Score threshold in the sidebar.")
            else:
                st.dataframe(anomalies, use_container_width=True)

        with tab_forecast:
            monthly = rolling_trend(filtered_state)[["Date", UNEMP_COL]]
            forecast = simple_forecast(monthly, periods_ahead=3)
            st.plotly_chart(forecast_chart(monthly, forecast), use_container_width=True)
            st.caption("⚠️ Linear forecast is illustrative only — not for policy decisions.")

            st.subheader("Statistical Test: Pre-COVID vs Lockdown")
            st.json({k: round(v, 4) if isinstance(v, float) else v for k, v in stats_result.items()})

    # ══════════════════════════════════════════
    # PAGE: Rural vs Urban
    # ══════════════════════════════════════════
    elif page == "🌾 Rural vs Urban":
        area_periods = period_averages(filtered_area, group_col="Area")
        pivot = area_periods.pivot(index="Area", columns="Period", values=UNEMP_COL).reset_index()
        st.plotly_chart(rural_urban_period_chart(pivot), use_container_width=True)

        area_trend = filtered_area.groupby(["Date", "Area"])[UNEMP_COL].mean().reset_index()
        fig = px.line(area_trend, x="Date", y=UNEMP_COL, color="Area", markers=True,
                      title="Rural vs Urban Over Time")
        fig.add_vrect(x0="2020-03-25", x1="2020-05-31", fillcolor="rgba(231,76,60,0.1)", line_width=0)
        st.plotly_chart(fig, use_container_width=True)

        lpr_trend = filtered_area.groupby(["Date", "Area"])[LPR_COL].mean().reset_index()
        fig2 = px.area(lpr_trend, x="Date", y=LPR_COL, color="Area",
                       title="Labour Participation — Rural vs Urban")
        st.plotly_chart(fig2, use_container_width=True)

    # ══════════════════════════════════════════
    # PAGE: Seasonal
    # ══════════════════════════════════════════
    elif page == "📅 Seasonal":
        seasonal = seasonal_decomposition_table(filtered_area)
        if seasonal.empty:
            st.info("No seasonal baseline data (2019) for the selected filters.")
        else:
            st.plotly_chart(seasonal_chart(seasonal), use_container_width=True)

            lpr_seasonal = seasonal.groupby("Month")["avg_lpr"].mean().reset_index()
            lpr_seasonal["MonthName"] = lpr_seasonal["Month"].map(
                lambda m: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][m-1]
            )
            fig = px.bar(lpr_seasonal, x="MonthName", y="avg_lpr", color="avg_lpr",
                         color_continuous_scale="Purples", title="Seasonal Labour Participation (2019)")
            st.plotly_chart(fig, use_container_width=True)

            peak = seasonal.groupby("Month")["avg_unemployment"].mean().idxmax()
            months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            st.info(f"Peak unemployment month (2019 baseline): **{months[peak-1]}**")

    # ══════════════════════════════════════════
    # PAGE: Policy & Export
    # ══════════════════════════════════════════
    elif page == "💡 Policy & Export":
        impact = state_lockdown_impact(filtered_state)
        worst = impact.nlargest(3, "unemp_change_pp")["Region"].tolist()
        best = impact.nsmallest(3, "recovery_gap_pp")["Region"].tolist()

        rural_pre = filtered_area[(filtered_area["Date"] < "2020-03-01") & (filtered_area["Area"] == "Rural")][UNEMP_COL].mean()
        urban_pre = filtered_area[(filtered_area["Date"] < "2020-03-01") & (filtered_area["Area"] == "Urban")][UNEMP_COL].mean()
        rural_lock = filtered_area[
            (filtered_area["Date"] >= "2020-04-01") & (filtered_area["Date"] <= "2020-05-31") & (filtered_area["Area"] == "Rural")
        ][UNEMP_COL].mean()
        urban_lock = filtered_area[
            (filtered_area["Date"] >= "2020-04-01") & (filtered_area["Date"] <= "2020-05-31") & (filtered_area["Area"] == "Urban")
        ][UNEMP_COL].mean()

        st.markdown(f"""
### Key Policy Insights

**1. COVID-19 Shock** — Unemployment rose from **{pre:.1f}%** to **{lock:.1f}%** ({lock-pre:.1f} pp).
Statistically significant at 95% confidence: **{'Yes' if stats_result['significant_95'] else 'No'}**.

**2. Rural–Urban Divide** — Pre-COVID Rural {rural_pre:.1f}% vs Urban {urban_pre:.1f}%.
Lockdown: Rural {rural_lock:.1f}% vs Urban {urban_lock:.1f}%.

**3. Regional Priority States** — Worst hit: **{', '.join(worst)}**. Best recovery: **{', '.join(best)}**.

**4. Recommendations**
- Expand MGNREGA & rural supply-chain support during shocks
- MSME credit + job retention schemes for urban areas
- State-targeted fiscal transfers based on recovery index
- Real-time rural/urban labour monitoring dashboards
- Portable social protection for migrant & informal workers
        """)

        st.divider()
        st.subheader("📥 Export Data")
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            st.download_button("Download State Data (CSV)", export_csv(filtered_state),
                               "unemployment_state_filtered.csv", "text/csv")
        with ec2:
            st.download_button("Download Rural/Urban Data (CSV)", export_csv(filtered_area),
                               "unemployment_area_filtered.csv", "text/csv")
        with ec3:
            st.download_button("Download Impact Analysis (CSV)", export_csv(impact),
                               "covid_impact_analysis.csv", "text/csv")

        with st.expander("View Raw Data"):
            choice = st.radio("Dataset", ["State-level", "Rural/Urban", "Impact Analysis"])
            if choice == "State-level":
                st.dataframe(filtered_state, use_container_width=True, height=400)
            elif choice == "Rural/Urban":
                st.dataframe(filtered_area, use_container_width=True, height=400)
            else:
                st.dataframe(impact, use_container_width=True, height=400)


if __name__ == "__main__":
    main()
