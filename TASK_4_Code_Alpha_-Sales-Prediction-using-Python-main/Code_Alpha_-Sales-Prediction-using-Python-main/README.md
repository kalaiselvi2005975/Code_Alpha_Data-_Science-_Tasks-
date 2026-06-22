# Sales Prediction using Python

This project builds an advanced sales prediction app using the dataset already available in the folder: `data/Advertising.csv`.

## What this project does

- Cleans and prepares the input dataset
- Detects the target column automatically
- Performs feature engineering such as:
  - total advertising spend
  - channel share features
  - interaction features
  - date-based features when a date column exists
- Compares multiple regression models:
  - Linear Regression
  - Ridge
  - Random Forest
  - Gradient Boosting
- Selects the best model based on cross-validated R²
- Saves trained artifacts and evaluation files
- Launches a Streamlit app for:
  - single prediction
  - batch forecasting
  - what-if advertising analysis
  - retraining with richer datasets

## Current dataset note

The included dataset contains these columns:

- `TV`
- `Radio`
- `Newspaper`
- `Sales`

It does **not** currently include `target segment`, `platform`, or `date/time` columns.  
The app is built to support those columns automatically if you upload a richer dataset later.

## Project files

- `app.py` : Streamlit application
- `train_model.py` : model training script
- `model_utils.py` : preprocessing, feature engineering, model training, artifact saving
- `requirements.txt` : Python dependencies
- `data/Advertising.csv` : input dataset
- `artifacts/` : generated after training

## Tech stack

This project uses the following tech stack:

- `Python` : core programming language
- `Pandas` : data loading, cleaning, transformation, and analysis
- `NumPy` : numerical operations
- `Scikit-learn` : preprocessing, feature engineering pipeline, model training, cross-validation, and evaluation
- `Streamlit` : interactive web app for predictions and scenario analysis
- `Plotly` : interactive charts and visual analytics
- `Joblib` : model artifact saving and loading
- `Matplotlib` and `Seaborn` : optional data visualization support for future analysis extensions

## Model stack

The machine learning layer currently compares multiple regression models:

- `Linear Regression`
- `Ridge Regression`
- `Random Forest Regressor`
- `Gradient Boosting Regressor`

The app automatically selects the best-performing model based on cross-validated `R²`.

## App architecture

- `Data layer` : CSV dataset input from `data/Advertising.csv`
- `Feature layer` : cleaned inputs, total ad spend, budget share, interaction features, and optional date features
- `Model layer` : training, evaluation, model selection, and saved artifacts
- `Presentation layer` : Streamlit dashboard with prediction, batch forecast, retraining, and scenario analysis

## Install

```bash
pip install -r requirements.txt
```

## Train the model

```bash
python train_model.py
```

## Run the app

```bash
streamlit run app.py
```

## Output files after training

The training script creates:

- `artifacts/sales_model.joblib`
- `artifacts/metrics.json`
- `artifacts/model_leaderboard.csv`
- `artifacts/feature_importance.csv`
- `artifacts/test_predictions.csv`
- `artifacts/business_insights.txt`

## Business value

- Helps estimate future sales before launching campaigns
- Shows how ad budget changes affect expected performance
- Highlights the most influential marketing drivers
- Supports smarter budget allocation decisions

## Recommended next upgrade

For a more advanced real-world solution, use a dataset with:

- campaign date
- target segment
- platform or channel group
- region
- seasonality factors
- promotion flags
- historical sales by period

That will allow time-series forecasting and more realistic marketing optimization.
