from pathlib import Path

import pandas as pd

from model_utils import detect_target, load_dataset, save_artifacts, train_best_model


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "data" / "Advertising.csv"
OUTPUT_DIR = BASE_DIR / "artifacts"


def build_business_insights(feature_importance: pd.DataFrame, metrics: dict) -> list[str]:
    insights = []
    top_features = feature_importance.head(5)["feature"].tolist()
    if top_features:
        insights.append(f"Top drivers of sales in the winning model: {', '.join(top_features)}.")

    insights.append(
        f"The best model is {metrics['best_model']} with test R² of {metrics['test_r2']:.3f}, "
        f"RMSE of {metrics['test_rmse']:.3f}, and MAE of {metrics['test_mae']:.3f}."
    )

    if any("radio" in feature.lower() for feature in top_features):
        insights.append("Radio spend appears highly influential, so campaign optimization should not focus only on TV budget.")
    if any("tv" in feature.lower() for feature in top_features):
        insights.append("TV remains a major volume driver, especially when combined with complementary channels.")
    if any("share" in feature.lower() for feature in top_features):
        insights.append("Budget mix matters, not only total spend. Rebalancing channel share can improve outcomes.")
    if any("x_" in feature.lower() for feature in top_features):
        insights.append("Channel interaction effects are present, suggesting multi-channel campaigns outperform isolated spending.")

    return insights


def main():
    df = load_dataset(str(DEFAULT_DATASET))
    target_col = detect_target(df, preferred_target="Sales")
    result = train_best_model(df, target_col=target_col)
    save_artifacts(result, str(OUTPUT_DIR))

    insights = build_business_insights(result["feature_importance"], result["metrics"])
    insight_path = OUTPUT_DIR / "business_insights.txt"
    insight_path.write_text("\n".join(f"- {item}" for item in insights), encoding="utf-8")

    print("Training complete")
    print(f"Best model: {result['metrics']['best_model']}")
    print(f"Test R2: {result['metrics']['test_r2']:.4f}")
    print(f"Artifacts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
