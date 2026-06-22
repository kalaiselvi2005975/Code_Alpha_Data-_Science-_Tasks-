"""Advanced analytics: period analysis, anomalies, correlations, rankings."""

import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    EMP_COL,
    LOCKDOWN_END,
    LOCKDOWN_START_MONTH,
    LPR_COL,
    PRE_COVID_END,
    RECOVERY_START,
    UNEMP_COL,
)


def filter_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    dates = df["Date"]
    if period == "pre":
        return df[dates < PRE_COVID_END]
    if period == "lock":
        return df[(dates >= LOCKDOWN_START_MONTH) & (dates <= LOCKDOWN_END)]
    if period == "recovery":
        return df[dates >= RECOVERY_START]
    return df


def period_averages(df: pd.DataFrame, group_col: str | None = None) -> pd.Series | pd.DataFrame:
    """Compute average unemployment across COVID periods."""
    periods = {
        "Pre-COVID": filter_by_period(df, "pre")[UNEMP_COL].mean(),
        "Lockdown": filter_by_period(df, "lock")[UNEMP_COL].mean(),
        "Recovery": filter_by_period(df, "recovery")[UNEMP_COL].mean(),
    }
    if group_col:
        rows = []
        for name, key in [("Pre-COVID", "pre"), ("Lockdown", "lock"), ("Recovery", "recovery")]:
            subset = filter_by_period(df, key)
            grouped = subset.groupby(group_col)[UNEMP_COL].mean()
            for idx, val in grouped.items():
                rows.append({"Period": name, group_col: idx, UNEMP_COL: val})
        return pd.DataFrame(rows)
    return pd.Series(periods)


def state_lockdown_impact(state_df: pd.DataFrame) -> pd.DataFrame:
    """State-level pre-COVID vs lockdown change with recovery metrics."""
    if state_df.empty:
        return pd.DataFrame(columns=[
            "Region", "pre_unemp", "pre_lpr", "pre_employed",
            "lock_unemp", "lock_lpr", "lock_employed",
            "recovery_unemp", "recovery_lpr",
            "unemp_change_pp", "recovery_gap_pp", "employment_loss_pct", "recovery_index",
        ])

    pre = filter_by_period(state_df, "pre").groupby("Region").agg(
        pre_unemp=(UNEMP_COL, "mean"),
        pre_lpr=(LPR_COL, "mean"),
        pre_employed=(EMP_COL, "mean"),
    )
    lock = filter_by_period(state_df, "lock").groupby("Region").agg(
        lock_unemp=(UNEMP_COL, "mean"),
        lock_lpr=(LPR_COL, "mean"),
        lock_employed=(EMP_COL, "mean"),
    )
    recovery = filter_by_period(state_df, "recovery").groupby("Region").agg(
        recovery_unemp=(UNEMP_COL, "mean"),
        recovery_lpr=(LPR_COL, "mean"),
    )

    result = pre.join(lock, how="inner").join(recovery, how="inner")
    result["unemp_change_pp"] = result["lock_unemp"] - result["pre_unemp"]
    result["recovery_gap_pp"] = result["recovery_unemp"] - result["pre_unemp"]
    result["employment_loss_pct"] = (
        (result["pre_employed"] - result["lock_employed"]) / result["pre_employed"] * 100
    )
    result["recovery_index"] = np.clip(
        100 - (result["recovery_gap_pp"] / result["unemp_change_pp"].replace(0, np.nan) * 100),
        0,
        100,
    )
    return result.sort_values("unemp_change_pp", ascending=False).reset_index()


def detect_anomalies(state_df: pd.DataFrame, z_threshold: float = 2.0) -> pd.DataFrame:
    """Flag months where state unemployment deviates significantly from its own mean."""
    columns = ["Region", "Date", UNEMP_COL, "Z-Score", "Type", "Zone"]
    if state_df.empty:
        return pd.DataFrame(columns=columns)

    records = []
    for region, group in state_df.groupby("Region"):
        mean = group[UNEMP_COL].mean()
        std = group[UNEMP_COL].std()
        if std == 0 or np.isnan(std):
            continue
        for _, row in group.iterrows():
            z = (row[UNEMP_COL] - mean) / std
            if abs(z) >= z_threshold:
                records.append({
                    "Region": region,
                    "Date": row["Date"],
                    UNEMP_COL: row[UNEMP_COL],
                    "Z-Score": round(z, 2),
                    "Type": "Spike" if z > 0 else "Dip",
                    "Zone": row.get("Zone", ""),
                })

    if not records:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(records).sort_values("Z-Score", ascending=False).reset_index(drop=True)


def correlation_matrix(state_df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation between labour market indicators."""
    cols = [UNEMP_COL, EMP_COL, LPR_COL]
    return state_df[cols].corr(method="pearson")


def rolling_trend(state_df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """National rolling average unemployment trend."""
    monthly = (
        state_df.groupby("Date")[UNEMP_COL]
        .mean()
        .reset_index()
        .sort_values("Date")
    )
    monthly["Rolling Mean"] = monthly[UNEMP_COL].rolling(window=window, min_periods=1).mean()
    monthly["Rolling Std"] = monthly[UNEMP_COL].rolling(window=window, min_periods=1).std().fillna(0)
    return monthly


def seasonal_decomposition_table(area_df: pd.DataFrame) -> pd.DataFrame:
    """Monthly seasonal averages from 2019 baseline."""
    baseline = area_df[area_df["Date"] < "2020-01-01"].copy()
    baseline["Month"] = baseline["Date"].dt.month
    return (
        baseline.groupby(["Month", "Area"])
        .agg(
            avg_unemployment=(UNEMP_COL, "mean"),
            avg_lpr=(LPR_COL, "mean"),
            avg_employed=(EMP_COL, "mean"),
        )
        .reset_index()
    )


def statistical_tests(state_df: pd.DataFrame) -> dict:
    """T-test: pre-COVID vs lockdown unemployment (national level)."""
    pre = filter_by_period(state_df, "pre")[UNEMP_COL].dropna()
    lock = filter_by_period(state_df, "lock")[UNEMP_COL].dropna()
    if len(pre) < 2 or len(lock) < 2:
        return {
            "pre_mean": pre.mean() if len(pre) else float("nan"),
            "lock_mean": lock.mean() if len(lock) else float("nan"),
            "t_statistic": float("nan"),
            "p_value": float("nan"),
            "significant_95": False,
        }
    t_stat, p_value = stats.ttest_ind(pre, lock, equal_var=False)
    return {
        "pre_mean": pre.mean(),
        "lock_mean": lock.mean(),
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant_95": p_value < 0.05,
    }


def top_bottom_states(state_df: pd.DataFrame, n: int = 5) -> dict:
    """Rank states by lockdown impact and recovery."""
    impact = state_lockdown_impact(state_df)
    return {
        "worst_lockdown": impact.nlargest(n, "unemp_change_pp")[["Region", "unemp_change_pp"]],
        "best_recovery": impact.nsmallest(n, "recovery_gap_pp")[["Region", "recovery_gap_pp"]],
        "highest_unemployment": (
            filter_by_period(state_df, "lock")
            .groupby("Region")[UNEMP_COL]
            .mean()
            .nlargest(n)
            .reset_index()
        ),
    }


def simple_forecast(monthly: pd.DataFrame, periods_ahead: int = 3) -> pd.DataFrame:
    """Linear trend extrapolation for next N months (educational/demo)."""
    y = monthly[UNEMP_COL].values
    x = np.arange(len(y))
    if len(y) < 2:
        return pd.DataFrame()
    slope, intercept = np.polyfit(x, y, 1)
    last_date = monthly["Date"].max()
    future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=periods_ahead, freq="MS")
    future_x = np.arange(len(y), len(y) + periods_ahead)
    forecast_vals = slope * future_x + intercept
    return pd.DataFrame({"Date": future_dates, UNEMP_COL: forecast_vals, "Type": "Forecast"})
