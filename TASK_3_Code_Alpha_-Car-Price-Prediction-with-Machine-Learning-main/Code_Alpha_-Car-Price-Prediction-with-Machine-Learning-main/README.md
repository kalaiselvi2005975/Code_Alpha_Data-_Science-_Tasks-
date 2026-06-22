# Car Price Prediction with Machine Learning

This project is a professional end-to-end machine learning solution for predicting used vehicle resale prices from structured tabular data. It covers data preprocessing, feature engineering, model training, evaluation, artifact generation, and deployment through a local Streamlit web app.

## Overview

The system uses the dataset in `data/car data.csv` and predicts the target variable `Selling_Price`. In addition to the original dataset columns, the project engineers practical business-oriented features such as vehicle age, brand goodwill, distance intensity, owner group, and price-age ratio to improve predictive quality.

The final workflow produces:

- A trained regression model saved to `models/car_price_model.joblib`
- Evaluation metrics in JSON and CSV format
- Visual charts for model comparison and error analysis
- A Streamlit application for interactive price prediction

## Tech stack

| Layer | Technology | Purpose |
|---|---|---|
| Programming language | `Python` | Core implementation |
| Data processing | `Pandas`, `NumPy` | Data loading, cleaning, feature engineering |
| Machine learning | `Scikit-learn` | Preprocessing pipeline, model training, evaluation |
| Visualization | `Matplotlib` | Performance and feature importance plots |
| Model persistence | `Joblib` | Saving and loading trained models |
| App interface | `Streamlit` | Local prediction dashboard |

## ML workflow

The project follows a complete supervised learning regression pipeline:

1. Load structured vehicle price data from CSV.
2. Create engineered features from raw business data.
3. Encode categorical fields and prepare numerical features.
4. Compare multiple ensemble regression models.
5. Select the best-performing model using cross-validation.
6. Evaluate final performance on a holdout test set.
7. Save the trained pipeline, metrics, and charts.
8. Serve predictions through a local app.

## Input features

### Original dataset columns

- `Car_Name`
- `Year`
- `Present_Price`
- `Driven_kms`
- `Fuel_Type`
- `Selling_type`
- `Transmission`
- `Owner`

### Engineered features

- `Brand`
- `Vehicle_Category`
- `Brand_Goodwill`
- `Vehicle_Age`
- `Log_Driven_kms`
- `Km_Per_Year`
- `Price_Age_Ratio`
- `Owner_Group`

## Models used

The training script compares the following regression algorithms:

- `RandomForestRegressor`
- `ExtraTreesRegressor`
- `GradientBoostingRegressor`

The best model is selected automatically based on cross-validated `RMSE`.

## Project structure

```text
TASK 3/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── car data.csv
├── models/
│   ├── car_price_model.joblib
│   └── model_metadata.json
└── reports/
    ├── actual_vs_predicted.png
    ├── feature_importance.csv
    ├── feature_importance.png
    ├── largest_errors.csv
    ├── model_comparison.csv
    ├── model_comparison.png
    ├── model_metrics.json
    └── residuals.png
```

## Installation

Open a terminal in the project folder and run:

```bash
python -m pip install -r requirements.txt
```

## How to run

### Step 1: Train the model

```bash
python train_model.py
```

This command will:

- Load the dataset
- Train and compare regression models
- Save the best model
- Generate metrics and plots inside `reports/`

### Step 2: Launch the app

```bash
python -m streamlit run app.py
```

After that, open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Expected output

After training, you should have:

- Saved model file in `models/`
- Metadata JSON file in `models/`
- Charts and evaluation reports in `reports/`
- A working Streamlit app for prediction

## Evaluation metrics

The project evaluates model quality using:

- `RMSE` : Root Mean Squared Error
- `MAE` : Mean Absolute Error
- `R²` : Goodness of fit
- `MAPE` : Mean Absolute Percentage Error

These metrics are saved in `reports/model_metrics.json`.

## Professional highlights

- Uses a preprocessing pipeline instead of manual scattered transformations
- Handles categorical and numerical data systematically
- Includes domain-style feature engineering for resale prediction
- Compares multiple ensemble regressors instead of training only one model
- Saves reusable model artifacts for deployment
- Provides a usable front-end application for testing predictions

## Troubleshooting

### If `python train_model.py` fails with a Matplotlib or `tkinter` error

The script has already been configured to use the non-GUI `Agg` backend, which is safe for Windows terminals.

### If Streamlit shows missing columns during prediction

The app has already been updated to generate the same engineered features used during model training.

### If the browser does not open automatically

Copy the local URL from the terminal and open it manually:

```text
http://localhost:8501
```

## Real-world applications

- Used car dealership pricing systems
- Vehicle resale marketplaces
- Trade-in recommendation tools
- Lending and insurance pre-valuation
- Fleet resale decision support

## Future improvements

For a more advanced version, you can extend the project with:

- Hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV`
- Better brand extraction with a lookup dictionary
- Additional features such as mileage, horsepower, engine size, location, and seller rating
- API deployment with `Flask` or `FastAPI`
- Cloud deployment for public access
