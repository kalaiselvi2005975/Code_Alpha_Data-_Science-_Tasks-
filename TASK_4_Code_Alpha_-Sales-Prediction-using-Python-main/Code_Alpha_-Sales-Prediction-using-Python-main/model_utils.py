import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


TARGET_CANDIDATES = ["sales", "revenue", "target", "y"]
DATE_CANDIDATES = ["date", "ds", "day", "month", "year", "timestamp"]


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [str(col).strip() for col in df.columns]

    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    return df


def detect_target(df: pd.DataFrame, preferred_target: str | None = None) -> str:
    if preferred_target and preferred_target in df.columns:
        return preferred_target

    lower_map = {col.lower(): col for col in df.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in lower_map:
            return lower_map[candidate]

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found. Please provide a target column.")
    return numeric_cols[-1]


def detect_date_column(df: pd.DataFrame) -> str | None:
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in DATE_CANDIDATES:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    lower_map = {col.lower(): col for col in data.columns}

    spend_cols = [
        col for col in numeric_cols
        if any(key in col.lower() for key in ["tv", "radio", "newspaper", "digital", "social", "search", "ad", "advert", "spend", "budget"])
    ]

    if spend_cols:
        data["total_ad_spend"] = data[spend_cols].sum(axis=1)
        for col in spend_cols:
            safe_total = data["total_ad_spend"].replace(0, np.nan)
            data[f"{col}_share"] = data[col] / safe_total
        if len(spend_cols) >= 2:
            first = spend_cols[0]
            second = spend_cols[1]
            data[f"{first}_x_{second}"] = data[first] * data[second]

    date_col = detect_date_column(data)
    if date_col:
        parsed = pd.to_datetime(data[date_col], errors="coerce")
        if parsed.notna().sum() > 0:
            data[date_col] = parsed
            data["year"] = parsed.dt.year
            data["month"] = parsed.dt.month
            data["quarter"] = parsed.dt.quarter
            data["dayofweek"] = parsed.dt.dayofweek
            data["is_month_start"] = parsed.dt.is_month_start.astype(int)
            data["is_month_end"] = parsed.dt.is_month_end.astype(int)
            data = data.drop(columns=[date_col])

    object_cols = data.select_dtypes(include=["object"]).columns.tolist()
    for col in object_cols:
        sample_non_null = data[col].dropna().astype(str)
        if sample_non_null.empty:
            continue
        if sample_non_null.nunique() > 0 and sample_non_null.nunique() < max(30, int(len(data) * 0.5)):
            data[col] = sample_non_null.reindex(data.index)

    return data


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def build_candidate_models(random_state: int = 42) -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
    }


def prepare_features_and_target(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    return X, y


def train_best_model(df: pd.DataFrame, target_col: str, random_state: int = 42) -> dict:
    X, y = prepare_features_and_target(df, target_col)
    engineered_sample = add_engineered_features(X)
    preprocessor = build_preprocessor(engineered_sample)
    models = build_candidate_models(random_state=random_state)

    results = []
    best_name = None
    best_cv_score = -np.inf
    best_pipeline = None

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("feature_engineering", FunctionTransformer(add_engineered_features, validate=False)),
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="r2",
        )
        mean_score = float(np.mean(scores))
        results.append(
            {
                "model": name,
                "cv_r2_mean": mean_score,
                "cv_r2_std": float(np.std(scores)),
            }
        )
        if mean_score > best_cv_score:
            best_cv_score = mean_score
            best_name = name
            best_pipeline = pipeline

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
    )

    best_pipeline.fit(X_train, y_train)
    predictions = best_pipeline.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae = float(mean_absolute_error(y_test, predictions))
    r2 = float(r2_score(y_test, predictions))

    feature_names = get_feature_names(best_pipeline.named_steps["preprocessor"])
    importances = extract_feature_importance(best_pipeline.named_steps["model"], feature_names)

    prediction_frame = X_test.copy()
    prediction_frame[target_col] = y_test.values
    prediction_frame["predicted_sales"] = predictions
    prediction_frame["absolute_error"] = np.abs(prediction_frame[target_col] - prediction_frame["predicted_sales"])

    return {
        "pipeline": best_pipeline,
        "leaderboard": pd.DataFrame(results).sort_values("cv_r2_mean", ascending=False).reset_index(drop=True),
        "metrics": {
            "best_model": best_name,
            "test_rmse": rmse,
            "test_mae": mae,
            "test_r2": r2,
            "rows": int(len(df)),
            "features_used": int(engineered_sample.shape[1]),
            "target_column": target_col,
        },
        "feature_importance": importances,
        "test_predictions": prediction_frame.reset_index(drop=True),
        "training_columns": X.columns.tolist(),
        "original_columns": df.columns.tolist(),
    }


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    if hasattr(preprocessor, "get_feature_names_out"):
        return list(preprocessor.get_feature_names_out())

    output_features = []
    for _, _, columns in preprocessor.transformers_:
        if isinstance(columns, list):
            output_features.extend(columns)
    return output_features


def extract_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = np.ravel(model.coef_)
        values = np.abs(coef)
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    df = pd.DataFrame({"feature": feature_names, "importance": values})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def save_artifacts(result: dict, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(result["pipeline"], output_path / "sales_model.joblib")
    result["leaderboard"].to_csv(output_path / "model_leaderboard.csv", index=False)
    result["feature_importance"].to_csv(output_path / "feature_importance.csv", index=False)
    result["test_predictions"].to_csv(output_path / "test_predictions.csv", index=False)

    with open(output_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result["metrics"], f, indent=2)


def load_artifacts(output_dir: str) -> tuple:
    output_path = Path(output_dir)
    pipeline = joblib.load(output_path / "sales_model.joblib")
    metrics = json.loads((output_path / "metrics.json").read_text(encoding="utf-8"))
    leaderboard = pd.read_csv(output_path / "model_leaderboard.csv")
    feature_importance = pd.read_csv(output_path / "feature_importance.csv")
    test_predictions = pd.read_csv(output_path / "test_predictions.csv")
    return pipeline, metrics, leaderboard, feature_importance, test_predictions


def apply_scenario_changes(df: pd.DataFrame, percent_changes: dict[str, float]) -> pd.DataFrame:
    changed = df.copy()
    for col, pct in percent_changes.items():
        if col in changed.columns:
            changed[col] = changed[col] * (1 + pct / 100.0)
    return changed
