"""Data loading and cleaning pipeline."""

import pandas as pd

from src.config import AREA_DATA_FILE, EMP_COL, LPR_COL, STATE_DATA_FILE, UNEMP_COL


def _clean_common(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"].str.strip(), format="%d-%m-%Y", errors="coerce")
    for col in [UNEMP_COL, EMP_COL, LPR_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Region"] = df["Region"].str.strip()
    return df


def load_state_data() -> pd.DataFrame:
    """Load state-level dataset with geographic zones (Jan–Oct 2020)."""
    df = pd.read_csv(STATE_DATA_FILE)
    df = _clean_common(df)
    df = df.rename(columns={"Region.1": "Zone"})
    for col in ["longitude", "latitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Zone"] = df["Zone"].str.strip()
    df = df.dropna(subset=["Date", UNEMP_COL])
    return df.sort_values(["Region", "Date"]).reset_index(drop=True)


def load_area_data() -> pd.DataFrame:
    """Load rural/urban breakdown dataset (May 2019 onwards)."""
    df = pd.read_csv(AREA_DATA_FILE)
    df = _clean_common(df)
    df["Area"] = df["Area"].str.strip()
    df = df.dropna(subset=["Date", UNEMP_COL, "Area"])
    return df.sort_values(["Region", "Area", "Date"]).reset_index(drop=True)


def merge_datasets(state_df: pd.DataFrame, area_df: pd.DataFrame) -> pd.DataFrame:
    """Combine state and area datasets for unified analytics."""
    state = state_df.copy()
    state["Area"] = "All"
    area = area_df.copy()
    if "Zone" not in area.columns:
        zone_map = state_df.drop_duplicates("Region").set_index("Region")["Zone"]
        area["Zone"] = area["Region"].map(zone_map)
    shared = [c for c in state.columns if c in area.columns]
    return pd.concat([state[shared], area[shared]], ignore_index=True)


def get_data_summary(state_df: pd.DataFrame, area_df: pd.DataFrame) -> dict:
    """Return high-level dataset metadata for the dashboard."""
    return {
        "state_records": len(state_df),
        "area_records": len(area_df),
        "states": state_df["Region"].nunique(),
        "state_date_min": state_df["Date"].min(),
        "state_date_max": state_df["Date"].max(),
        "area_date_min": area_df["Date"].min(),
        "area_date_max": area_df["Date"].max(),
        "zones": sorted(state_df["Zone"].unique()),
    }
