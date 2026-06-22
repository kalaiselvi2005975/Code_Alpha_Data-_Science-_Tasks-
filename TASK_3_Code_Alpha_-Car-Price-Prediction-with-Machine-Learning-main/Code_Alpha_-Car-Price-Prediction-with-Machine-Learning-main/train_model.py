from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "car data.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
CURRENT_YEAR = 2026
RANDOM_STATE = 42


def infer_brand(vehicle_name: str) -> str:
    text = str(vehicle_name).strip().lower()
    keyword_map = {
        "Maruti": [
            "ritz",
            "sx4",
            "ciaz",
            "wagon r",
            "wagon",
            "swift",
            "vitara",
            "ertiga",
            "dzire",
            "alto",
            "ignis",
            "baleno",
            "omni",
            "s cross",
            "800",
        ],
        "Honda": ["city", "amaze", "brio", "jazz", "activa"],
        "Toyota": ["corolla", "innova", "fortuner", "etios", "camry", "land cruiser"],
        "Hyundai": ["verna", "elantra", "creta", "eon", "i10", "i20", "xcent", "grand"],
        "Mahindra": ["mahindra"],
        "Bajaj": ["bajaj"],
        "Hero": ["hero"],
        "Yamaha": ["yamaha"],
        "TVS": ["tvs"],
        "KTM": ["ktm"],
        "Royal Enfield": ["royal"],
        "Suzuki": ["suzuki"],
        "UM": ["um"],
        "Hyosung": ["hyosung"],
    }

    for brand, keywords in keyword_map.items():
        if any(keyword in text for keyword in keywords):
            return brand
    return text.split()[0].title() if text else "Unknown"


def infer_vehicle_category(brand: str) -> str:
    bike_brands = {"Bajaj", "Hero", "Yamaha", "TVS", "KTM", "Royal Enfield", "Suzuki", "UM", "Hyosung"}
    return "Two Wheeler" if brand in bike_brands else "Four Wheeler"


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["Brand"] = data["Car_Name"].apply(infer_brand)
    data["Vehicle_Category"] = data["Brand"].apply(infer_vehicle_category)

    goodwill_score = {
        "Toyota": 5,
        "Honda": 5,
        "Hyundai": 4,
        "Maruti": 4,
        "Mahindra": 4,
        "Royal Enfield": 4,
        "KTM": 4,
        "Bajaj": 3,
        "Hero": 3,
        "Yamaha": 4,
        "TVS": 3,
        "Suzuki": 4,
        "UM": 2,
        "Hyosung": 2,
    }

    data["Brand_Goodwill"] = data["Brand"].map(goodwill_score).fillna(3)
    data["Vehicle_Age"] = (CURRENT_YEAR - data["Year"]).clip(lower=0)
    data["Log_Driven_kms"] = np.log1p(data["Driven_kms"])
    data["Km_Per_Year"] = data["Driven_kms"] / data["Vehicle_Age"].replace(0, 1)
    data["Price_Age_Ratio"] = data["Present_Price"] / (data["Vehicle_Age"] + 1)
    data["Owner_Group"] = np.where(data["Owner"] <= 1, "Low", np.where(data["Owner"] == 2, "Medium", "High"))

    return data


def create_preprocessor(feature_frame: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    categorical_cols = [
        "Car_Name",
        "Fuel_Type",
        "Selling_type",
        "Transmission",
        "Brand",
        "Vehicle_Category",
        "Owner_Group",
    ]
    numeric_cols = [
        "Year",
        "Present_Price",
        "Driven_kms",
        "Owner",
        "Brand_Goodwill",
        "Vehicle_Age",
        "Log_Driven_kms",
        "Km_Per_Year",
        "Price_Age_Ratio",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            ),
        ]
    )

    return preprocessor, categorical_cols, numeric_cols


def evaluate_models(X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer) -> pd.DataFrame:
    candidate_models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=450,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=450,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=350,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for name, model in candidate_models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "r2": "r2",
            },
            n_jobs=-1,
        )

        rows.append(
            {
                "model": name,
                "cv_rmse_mean": float(-cv_scores["test_rmse"].mean()),
                "cv_rmse_std": float(cv_scores["test_rmse"].std()),
                "cv_mae_mean": float(-cv_scores["test_mae"].mean()),
                "cv_r2_mean": float(cv_scores["test_r2"].mean()),
            }
        )

    results = pd.DataFrame(rows).sort_values("cv_rmse_mean").reset_index(drop=True)
    return results


def get_model_by_name(name: str):
    model_map = {
        "RandomForest": RandomForestRegressor(
            n_estimators=450,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=450,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=350,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }
    return model_map[name]


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-9, None))) * 100
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_percent": float(mape),
    }


def save_plot_model_comparison(results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(results["model"], results["cv_rmse_mean"], color=["#2563eb", "#16a34a", "#dc2626"])
    ax.set_title("Cross-validated RMSE by model")
    ax.set_ylabel("RMSE")
    ax.set_xlabel("Model")
    for idx, value in enumerate(results["cv_rmse_mean"]):
        ax.text(idx, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "model_comparison.png", dpi=180)
    plt.close(fig)


def save_plot_actual_vs_predicted(y_test: pd.Series, y_pred: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.75, color="#2563eb")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="#dc2626")
    ax.set_title("Actual vs Predicted Prices")
    ax.set_xlabel("Actual Selling Price")
    ax.set_ylabel("Predicted Selling Price")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "actual_vs_predicted.png", dpi=180)
    plt.close(fig)


def save_plot_residuals(y_test: pd.Series, y_pred: np.ndarray) -> None:
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals, alpha=0.75, color="#16a34a")
    ax.axhline(0, color="#dc2626", linestyle="--")
    ax.set_title("Residual Plot")
    ax.set_xlabel("Predicted Selling Price")
    ax.set_ylabel("Residual")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "residuals.png", dpi=180)
    plt.close(fig)


def save_feature_importance(best_pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = best_pipeline.named_steps["preprocessor"]
    regressor = best_pipeline.named_steps["regressor"]

    feature_names = preprocessor.get_feature_names_out()
    if not hasattr(regressor, "feature_importances_"):
        importance_df = pd.DataFrame({"feature": feature_names, "importance": np.nan})
        importance_df.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
        return importance_df

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": regressor.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)

    top_features = importance_df.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_features["feature"], top_features["importance"], color="#7c3aed")
    ax.set_title("Top 15 Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "feature_importance.png", dpi=180)
    plt.close(fig)
    return importance_df


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    model_df = build_feature_frame(df)

    target_column = "Selling_Price"
    X = model_df.drop(columns=[target_column])
    y = model_df[target_column]

    preprocessor, categorical_cols, numeric_cols = create_preprocessor(model_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    comparison = evaluate_models(X_train, y_train, preprocessor)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    save_plot_model_comparison(comparison)

    best_model_name = comparison.loc[0, "model"]
    best_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", get_model_by_name(best_model_name)),
        ]
    )
    best_pipeline.fit(X_train, y_train)

    y_pred = best_pipeline.predict(X_test)
    holdout_metrics = regression_metrics(y_test, y_pred)
    holdout_metrics["best_model"] = best_model_name
    holdout_metrics["train_rows"] = int(len(X_train))
    holdout_metrics["test_rows"] = int(len(X_test))
    holdout_metrics["dataset_rows"] = int(len(df))

    save_plot_actual_vs_predicted(y_test, y_pred)
    save_plot_residuals(y_test, y_pred)
    importance_df = save_feature_importance(best_pipeline)

    preview_predictions = X_test.copy()
    preview_predictions["actual_price"] = y_test.values
    preview_predictions["predicted_price"] = y_pred
    preview_predictions["abs_error"] = np.abs(preview_predictions["actual_price"] - preview_predictions["predicted_price"])
    preview_predictions.sort_values("abs_error", ascending=False).head(15).to_csv(
        REPORTS_DIR / "largest_errors.csv", index=False
    )

    bundle = {
        "model": best_pipeline,
        "feature_columns": X.columns.tolist(),
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
        "current_year": CURRENT_YEAR,
        "model_name": best_model_name,
        "holdout_metrics": holdout_metrics,
        "top_features": importance_df.head(10).to_dict(orient="records"),
    }
    joblib.dump(bundle, MODELS_DIR / "car_price_model.joblib")

    metadata = {
        "dataset_path": str(DATA_PATH),
        "target": target_column,
        "best_model": best_model_name,
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "engineered_features": [
            "Brand",
            "Vehicle_Category",
            "Brand_Goodwill",
            "Vehicle_Age",
            "Log_Driven_kms",
            "Km_Per_Year",
            "Price_Age_Ratio",
            "Owner_Group",
        ],
        "holdout_metrics": holdout_metrics,
    }

    with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(REPORTS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Training complete.")
    print(f"Best model: {best_model_name}")
    print(json.dumps(holdout_metrics, indent=2))


if __name__ == "__main__":
    main()
