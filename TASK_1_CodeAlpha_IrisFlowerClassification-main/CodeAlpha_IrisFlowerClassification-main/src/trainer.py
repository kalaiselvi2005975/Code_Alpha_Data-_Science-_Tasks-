from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from src.config import (
    FEATURE_COLUMNS,
    KNN_NEIGHBORS,
    MODEL_PATH,
    OUTPUT_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data_loader import load_data


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> KNeighborsClassifier:
    model = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: KNeighborsClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path = OUTPUT_DIR,
) -> float:
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    print(f"\nAccuracy: {accuracy:.2%}\n")
    print("Classification Report:")
    print(classification_report(y_test, predictions))

    cm = confusion_matrix(y_test, predictions, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix - Iris Classification")
    plt.tight_layout()

    plot_path = output_dir / "confusion_matrix.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"Confusion matrix saved to {plot_path}")

    return accuracy


def save_model(model: KNeighborsClassifier, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path: Path = MODEL_PATH) -> KNeighborsClassifier:
    return joblib.load(path)


def predict(
    model: KNeighborsClassifier,
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
) -> str:
    sample = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=FEATURE_COLUMNS,
    )
    return model.predict(sample)[0]


def run_training() -> KNeighborsClassifier:
    print("Loading Iris dataset...")
    X, y = load_data()
    print(f"Loaded {len(X)} samples with {X.shape[1]} features.\n")
    print("Feature columns:", ", ".join(FEATURE_COLUMNS))
    print("Species classes:", y.unique().tolist(), "\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}\n")

    print("Training K-Nearest Neighbors classifier...")
    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)
    save_model(model)

    example_species = predict(model, 5.9, 3.0, 4.2, 1.5)
    print("\nExample prediction:")
    print("  Measurements: sepal 5.9x3.0 cm, petal 4.2x1.5 cm")
    print(f"  Predicted species: {example_species}")

    return model
