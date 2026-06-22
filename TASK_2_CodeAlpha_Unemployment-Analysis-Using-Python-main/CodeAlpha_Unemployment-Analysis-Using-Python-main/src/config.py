"""Project configuration and constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "DATA"
OUTPUT_DIR = PROJECT_ROOT / "output"

STATE_DATA_FILE = DATA_DIR / "Unemployment_Rate_upto_11_2020.csv"
AREA_DATA_FILE = DATA_DIR / "Unemployment in India.csv"

LOCKDOWN_START = "2020-03-25"
LOCKDOWN_END = "2020-05-31"
PRE_COVID_END = "2020-03-01"
LOCKDOWN_START_MONTH = "2020-04-01"
RECOVERY_START = "2020-06-01"

UNEMP_COL = "Estimated Unemployment Rate (%)"
EMP_COL = "Estimated Employed"
LPR_COL = "Estimated Labour Participation Rate (%)"

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

PERIODS = {
    "Pre-COVID (Jan–Feb 2020)": ("pre", PRE_COVID_END),
    "Lockdown (Apr–May 2020)": ("lock", LOCKDOWN_START_MONTH, LOCKDOWN_END),
    "Recovery (Jun–Oct 2020)": ("recovery", RECOVERY_START),
}

COLOR_PRE = "#2ecc71"
COLOR_LOCK = "#e74c3c"
COLOR_RECOVERY = "#3498db"
