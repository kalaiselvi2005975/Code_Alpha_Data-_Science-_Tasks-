from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Iris.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "iris_model.joblib"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

FEATURE_COLUMNS = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]

TEST_SIZE = 0.2
RANDOM_STATE = 42
KNN_NEIGHBORS = 3
