from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model_utils import apply_scenario_changes, detect_target, load_artifacts, load_dataset, save_artifacts, train_best_model


st.set_page_config(page_title="Advanced Sales Prediction App", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "data" / "Advertising.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"


@st.cache_data
def get_default_data():
    return load_dataset(str(DEFAULT_DATASET))


@st.cache_data
def get_trained_artifacts():
    return load_artifacts(str(ARTIFACT_DIR))


def ensure_artifacts():
    if not ARTIFACT_DIR.exists():
        df = get_default_data()
        target_col = detect_target(df, preferred_target="Sales")
        result = train_best_model(df, target_col=target_col)
        save_artifacts(result, str(ARTIFACT_DIR))


def prediction_input_form(reference_df: pd.DataFrame):
    st.subheader("Single campaign prediction")
    input_payload = {}

    with st.form("predict_form"):
        cols = st.columns(3)
        numeric_features = reference_df.select_dtypes(include="number").columns.tolist()
        target_col = detect_target(reference_df, preferred_target="Sales")
        numeric_features = [c for c in numeric_features if c != target_col]

        object_features = reference_df.select_dtypes(exclude="number").columns.tolist()
        object_features = [c for c in object_features if c != target_col]

        for idx, col in enumerate(numeric_features):
            default_val = float(reference_df[col].median())
            input_payload[col] = cols[idx % 3].number_input(
                col,
                min_value=0.0,
                value=default_val,
                step=1.0,
            )

        for idx, col in enumerate(object_features):
            choices = sorted(reference_df[col].dropna().astype(str).unique().tolist())
            if choices:
                input_payload[col] = cols[idx % 3].selectbox(col, choices)

        submitted = st.form_submit_button("Predict sales")

    if submitted:
        return pd.DataFrame([input_payload])
    return None


def scenario_controls(reference_df: pd.DataFrame):
    st.subheader("Advertising scenario simulator")
    numeric_cols = reference_df.select_dtypes(include="number").columns.tolist()
    target_col = detect_target(reference_df, preferred_target="Sales")
    spend_cols = [
        col for col in numeric_cols
        if col != target_col and any(key in col.lower() for key in ["tv", "radio", "newspaper", "digital", "social", "search", "ad", "advert", "spend", "budget"])
    ]

    if not spend_cols:
        st.info("No advertising spend columns were detected for scenario simulation.")
        return {}

    changes = {}
    cols = st.columns(min(3, len(spend_cols)))
    for idx, col in enumerate(spend_cols):
        changes[col] = cols[idx % len(cols)].slider(
            f"{col} change %",
            min_value=-50,
            max_value=100,
            value=0,
            step=5,
        )
    return changes


def main():
    ensure_artifacts()
    pipeline, metrics, leaderboard, feature_importance, test_predictions = get_trained_artifacts()
    default_df = get_default_data()
    target_col = detect_target(default_df, preferred_target="Sales")

    st.title("Advanced Sales Prediction and Marketing Insights")
    st.caption("Regression forecasting app with feature engineering, model comparison, scenario analysis, and batch prediction support.")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Best model", metrics["best_model"])
    kpi2.metric("Test R²", f"{metrics['test_r2']:.3f}")
    kpi3.metric("RMSE", f"{metrics['test_rmse']:.3f}")
    kpi4.metric("Features used", metrics["features_used"])

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Overview", "Predict", "Scenario Analysis", "Batch Forecast", "Retrain"]
    )

    with tab1:
        left, right = st.columns([1.1, 1])
        with left:
            st.subheader("Dataset preview")
            st.dataframe(default_df.head(20), use_container_width=True)

            st.subheader("Model leaderboard")
            st.dataframe(leaderboard, use_container_width=True)

        with right:
            st.subheader("Feature importance")
            if not feature_importance.empty:
                fig = px.bar(
                    feature_importance.head(12),
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Top model drivers",
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("This model does not expose feature importance.")

            st.subheader("Actual vs predicted")
            fig2 = px.scatter(
                test_predictions,
                x=target_col,
                y="predicted_sales",
                title="Prediction quality on holdout data",
            )
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        user_input_df = prediction_input_form(default_df)
        if user_input_df is not None:
            pred = pipeline.predict(user_input_df)[0]
            st.success(f"Predicted sales: {pred:.2f}")

            if target_col in default_df.columns:
                baseline_mean = default_df[target_col].mean()
                delta = pred - baseline_mean
                st.write(f"Difference from average sales: {delta:+.2f}")

    with tab3:
        changes = scenario_controls(default_df)
        if changes:
            scenario_df = apply_scenario_changes(default_df.drop(columns=[target_col]), changes)
            base_pred = pipeline.predict(default_df.drop(columns=[target_col]))
            scenario_pred = pipeline.predict(scenario_df)

            summary = pd.DataFrame(
                {
                    "Scenario": ["Baseline", "Changed budget mix"],
                    "Average predicted sales": [base_pred.mean(), scenario_pred.mean()],
                }
            )
            st.dataframe(summary, use_container_width=True)

            fig3 = go.Figure()
            fig3.add_trace(go.Bar(name="Baseline", x=["Avg predicted sales"], y=[base_pred.mean()]))
            fig3.add_trace(go.Bar(name="Scenario", x=["Avg predicted sales"], y=[scenario_pred.mean()]))
            fig3.update_layout(barmode="group", title="Impact of ad-spend changes")
            st.plotly_chart(fig3, use_container_width=True)

            uplift = scenario_pred.mean() - base_pred.mean()
            uplift_pct = (uplift / base_pred.mean()) * 100 if base_pred.mean() else 0
            st.info(f"Estimated average sales impact: {uplift:+.2f} ({uplift_pct:+.2f}%)")

    with tab4:
        st.subheader("Upload future campaign plan")
        uploaded = st.file_uploader("Upload CSV for batch forecasting", type=["csv"])
        if uploaded is not None:
            future_df = pd.read_csv(uploaded)
            predictions = pipeline.predict(future_df)
            forecast_df = future_df.copy()
            forecast_df["predicted_sales"] = predictions
            st.dataframe(forecast_df.head(50), use_container_width=True)
            st.download_button(
                "Download predictions",
                data=forecast_df.to_csv(index=False).encode("utf-8"),
                file_name="sales_forecast_predictions.csv",
                mime="text/csv",
            )

    with tab5:
        st.subheader("Retrain model with a richer dataset")
        st.write("Upload a CSV with optional date, segment, or platform columns. The app will automatically use them if present.")
        retrain_file = st.file_uploader("Upload training dataset", type=["csv"], key="retrain")
        if retrain_file is not None:
            new_df = pd.read_csv(retrain_file)
            target_guess = detect_target(new_df)
            result = train_best_model(new_df, target_col=target_guess)
            save_artifacts(result, str(ARTIFACT_DIR))
            st.success(
                f"Model retrained successfully. Best model: {result['metrics']['best_model']} | Test R²: {result['metrics']['test_r2']:.3f}"
            )
            st.dataframe(result["leaderboard"], use_container_width=True)

    with st.sidebar:
        st.header("Marketing guidance")
        st.write(
            "- Use the model leaderboard to compare algorithms.\n"
            "- Focus on the strongest drivers in feature importance.\n"
            "- Use scenario analysis before shifting budget across channels.\n"
            "- Upload future campaign plans for fast sales forecasting.\n"
            "- If you have date, segment, or platform data, retrain for a richer model."
        )


if __name__ == "__main__":
    main()
