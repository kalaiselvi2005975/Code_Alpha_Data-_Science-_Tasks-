# Unemployment Analysis — India

**Advanced Data Analytics & Interactive Dashboard**

A professional end-to-end Python project for analyzing unemployment rate data across Indian states and union territories. Includes data cleaning, exploratory analysis, COVID-19 impact assessment, seasonal trend detection, statistical testing, and an interactive Streamlit dashboard with policy-oriented insights.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Run Methods](#run-methods)
- [Dashboard Guide](#dashboard-guide)
- [Key Findings](#key-findings)
- [Datasets](#datasets)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Project Overview

This project fulfills **Task 2: Unemployment Analysis with Python** and extends it with an advanced, production-style analytics pipeline.

| Objective | Status |
|-----------|--------|
| Analyze unemployment rate data (percentage of unemployed people) | Done |
| Data cleaning, exploration, and visualization | Done |
| Investigate COVID-19 impact on unemployment (Mar–May 2020 lockdown) | Done |
| Identify seasonal trends (rural & urban) | Done |
| Present insights for economic and social policy | Done |
| Interactive web dashboard | Done |

---

## Tech Stack

### Core Stack

| Category | Technology | Version | Role in Project |
|----------|------------|---------|-----------------|
| **Programming Language** | Python | 3.10+ | Backend logic, analytics, and UI |
| **Data Manipulation** | Pandas | ≥ 2.0 | CSV loading, cleaning, filtering, aggregation |
| **Numerical Computing** | NumPy | ≥ 1.24 | Array math, forecasting, recovery index |
| **Statistics** | SciPy | ≥ 1.11 | Welch's t-test, statistical significance |
| **Static Charts** | Matplotlib | ≥ 3.7 | PNG chart export (batch mode) |
| **Statistical Plots** | Seaborn | ≥ 0.13 | Heatmaps, styled visualizations |
| **Interactive Charts** | Plotly | ≥ 5.18 | Line charts, geo maps, correlation heatmaps |
| **Web Framework** | Streamlit | ≥ 1.28 | Interactive dashboard UI |
| **File Export** | openpyxl | ≥ 3.1 | CSV / Excel export support |

### Architecture Pattern

| Pattern | Implementation |
|---------|----------------|
| **Project Type** | Data Analytics + Interactive Dashboard |
| **Data Storage** | Local CSV files (no database) |
| **Frontend** | Streamlit (Python-native, no React/HTML required) |
| **Backend** | Python modular package (`src/`) |
| **Deployment** | Local execution via Streamlit server |

### System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   PRESENTATION LAYER                          │
│   streamlit_app.py          unemployment_analysis.py          │
│   (Interactive Dashboard)   (Batch Script → PNG output/)      │
└─────────────────────────────┬────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────┐
│                   ANALYTICS LAYER                               │
│   src/analytics.py  — COVID impact, anomalies, t-test, forecast│
│   src/charts.py     — Plotly chart builders                     │
└─────────────────────────────┬────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────┐
│                   DATA LAYER                                    │
│   src/data_loader.py — Load, clean, validate CSV data           │
│   src/config.py      — Paths, column names, COVID period dates  │
└─────────────────────────────┬────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────┐
│                   DATA SOURCE (CSV)                             │
│   Unemployment_Rate_upto_11_2020.csv                          │
│   Unemployment in India.csv                                    │
└──────────────────────────────────────────────────────────────┘
```

### Data Pipeline Flow

```
CSV Files → Pandas (Clean) → Analytics (NumPy/SciPy) → Charts (Plotly/Matplotlib) → Streamlit UI
```

---

## Features

### Interactive Dashboard (7 Pages)

| Page | Capabilities |
|------|-------------|
| **Overview** | KPI metrics, national trend, dataset summary, state rankings |
| **Trends & Maps** | State comparison, geographic bubble map, zone trends, heatmap |
| **COVID Analysis** | Pre-COVID vs lockdown vs recovery, state impact table |
| **Advanced Analytics** | Correlation matrix, anomaly detection, linear forecast, t-test |
| **Rural vs Urban** | Area-wise COVID impact, labour participation trends |
| **Seasonal** | 2019 monthly baseline patterns |
| **Policy & Export** | Policy recommendations, CSV download buttons |

### Analytics Capabilities

- COVID-19 period comparison (Pre-COVID / Lockdown / Recovery)
- Z-score anomaly detection per state
- Pearson correlation matrix (unemployment, employment, participation)
- Welch's t-test for statistical significance
- State-level recovery index
- Geographic unemployment map (latitude/longitude)
- Seasonal decomposition (2019 baseline)
- Linear trend forecasting (illustrative)
- Filterable exports (State, Zone, Rural/Urban)

---

## Project Structure

```
TASK 2/
│
├── DATA/                                    # Raw datasets
│   ├── Unemployment_Rate_upto_11_2020.csv   # State-level (Jan–Oct 2020)
│   └── Unemployment in India.csv            # Rural/Urban (May 2019–Jun 2020)
│
├── src/                                     # Core analytics package
│   ├── __init__.py
│   ├── config.py                            # Constants & configuration
│   ├── data_loader.py                       # Data ingestion & cleaning
│   ├── analytics.py                         # Statistical & COVID analytics
│   └── charts.py                            # Plotly visualization builders
│
├── output/                                  # Generated PNG charts (batch mode)
│
├── streamlit_app.py                         # Main interactive dashboard
├── unemployment_analysis.py                 # Batch analysis script
├── test_app.py                              # Smoke test (verify no errors)
├── requirements.txt                         # Python dependencies
└── README.md                                # Project documentation
```

---

## Prerequisites

Before running the project, ensure you have:

| Requirement | Details |
|-------------|---------|
| **OS** | Windows 10/11, macOS, or Linux |
| **Python** | 3.10 or higher (tested on 3.13) |
| **pip** | Python package manager |
| **Browser** | Chrome, Edge, or Firefox (for dashboard) |
| **Terminal** | Command Prompt, PowerShell, or Terminal |

Verify Python is installed:

```cmd
python --version
```

Expected output: `Python 3.10.x` or higher.

---

## Installation

### Step 1 — Open terminal and navigate to project folder

**Windows (Command Prompt / PowerShell):**

```cmd
cd "C:\Users\Asus\Downloads\TASK 2"
```

**macOS / Linux:**

```bash
cd /path/to/TASK\ 2
```

### Step 2 — Install dependencies

> **Note (Windows):** If `pip install` fails with a launcher error, use `python -m pip` instead.

```cmd
python -m pip install -r requirements.txt
```

### Step 3 — Verify installation

```cmd
python -m pip list
```

You should see: `pandas`, `numpy`, `streamlit`, `plotly`, `scipy`, `matplotlib`, `seaborn`, `openpyxl`.

---

## Run Methods

### Method 1 — Interactive Dashboard (Recommended)

Launch the Streamlit web application:

```cmd
cd "C:\Users\Asus\Downloads\TASK 2"
python -m streamlit run streamlit_app.py
```

| Detail | Value |
|--------|-------|
| **URL** | http://localhost:8501 |
| **Stop server** | Press `Ctrl + C` in terminal |
| **Refresh app** | Press `R` in browser after code changes |

**Run on a different port:**

```cmd
python -m streamlit run streamlit_app.py --server.port 8502
```

Then open: http://localhost:8502

---

### Method 2 — Batch Analysis (PNG Charts)

Generate static charts and print insights to the terminal:

```cmd
cd "C:\Users\Asus\Downloads\TASK 2"
python unemployment_analysis.py
```

| Output | Location |
|--------|----------|
| PNG charts | `output/` folder |
| Console report | Terminal (exploration summary + policy insights) |

Generated files:

| File | Description |
|------|-------------|
| `01_national_trend.png` | National unemployment trend |
| `02_covid_impact.png` | COVID period comparison |
| `04_seasonal_patterns.png` | Seasonal unemployment pattern |
| `06_state_heatmap.png` | State × month heatmap |
| `07_zone_trends.png` | Zone-wise trends |
| `08_employment_scatter.png` | Employment vs unemployment |

---

### Method 3 — Run Tests (Verify No Errors)

Run the smoke test to validate all modules and chart functions:

```cmd
cd "C:\Users\Asus\Downloads\TASK 2"
python test_app.py
```

Expected output:

```
ALL TESTS PASSED - No errors
```

---

### Quick Reference — All Commands

| Action | Command |
|--------|---------|
| Install packages | `python -m pip install -r requirements.txt` |
| Run dashboard | `python -m streamlit run streamlit_app.py` |
| Run batch analysis | `python unemployment_analysis.py` |
| Run tests | `python test_app.py` |
| Upgrade pip | `python -m pip install --upgrade pip` |
| Stop dashboard | `Ctrl + C` |

---

## Dashboard Guide

### Sidebar Controls

| Control | Purpose |
|---------|---------|
| **Navigate** | Switch between 7 analysis pages |
| **State/UT** | Filter by state or union territory |
| **Zone** | Filter by geographic zone (North, South, East, West, Northeast) |
| **Area Type** | Filter Rural / Urban data |
| **Anomaly Z-Score Threshold** | Adjust sensitivity for anomaly detection (default: 2.0) |

### Recommended Workflow

1. Start with **Overview** for KPIs and national trend
2. Open **Trends & Maps → Geo Map** for geographic view
3. Review **COVID Analysis** for lockdown impact
4. Explore **Advanced Analytics** for correlation and anomalies
5. Compare **Rural vs Urban** and **Seasonal** patterns
6. Export data from **Policy & Export**

---

## Key Findings

| Metric | Value |
|--------|-------|
| Pre-COVID unemployment (avg) | ~9.2% |
| Lockdown peak (Apr–May 2020) | ~22.7% |
| Recovery period (Jun–Oct 2020) | ~9.6% |
| Relative increase during lockdown | ~146% |
| Highest impact zone | North (~15.9% avg) |
| Peak seasonal month (2019) | October |

### Policy Recommendations

1. Counter-cyclical employment programs (e.g. MGNREGA) during economic shocks
2. State-targeted fiscal transfers for high-impact, slow-recovery regions
3. Dual rural–urban strategy: agricultural supply chains + urban MSME support
4. Track labour participation alongside unemployment rate
5. Real-time disaggregated monitoring by state and area type
6. Portable social protection for informal and migrant workers

---

## Datasets

| File | Records | Coverage | Key Columns |
|------|---------|----------|-------------|
| `Unemployment_Rate_upto_11_2020.csv` | 267 | 27 states/UTs, Jan–Oct 2020 | Region, Date, Unemployment Rate, Employed, Labour Participation, Zone, Lat/Long |
| `Unemployment in India.csv` | 740 | 28 regions, May 2019–Jun 2020 | Region, Date, Unemployment Rate, Employed, Labour Participation, Area (Rural/Urban) |

Both files are located in the `DATA/` folder.

---

## Troubleshooting

### `pip install` fails — "Unable to create process" (Windows)

**Cause:** `pip` is linked to an old/uninstalled Python version.

**Fix:** Use `python -m pip` instead of `pip`:

```cmd
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

---

### `streamlit` command not found

**Fix:**

```cmd
python -m streamlit run streamlit_app.py
```

---

### Dashboard shows "No data matches your filters"

**Fix:** In the sidebar, select at least one **State/UT** and one **Zone**.

---

### Anomalies page shows "No anomalies detected"

**Fix:** Lower the **Anomaly Z-Score Threshold** slider to **2.0** in the sidebar.

---

### Port 8501 already in use

**Fix:** Run on another port:

```cmd
python -m streamlit run streamlit_app.py --server.port 8502
```

---

### Module not found error

**Fix:** Install from project root:

```cmd
cd "C:\Users\Asus\Downloads\TASK 2"
python -m pip install -r requirements.txt
```

---

## License

This project is developed for **educational and analytical purposes**.

---

## Author

**Task 2 — Unemployment Analysis with Python**  
Advanced Edition · Interactive Streamlit Dashboard · Modular Analytics Architecture

---

**One-line summary:** Python + Pandas + Plotly + Streamlit for unemployment data analysis and an interactive policy dashboard.
