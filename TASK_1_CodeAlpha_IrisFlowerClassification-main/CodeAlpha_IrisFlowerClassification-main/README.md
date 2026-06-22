# Iris Flower Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)


An end-to-end machine learning application that classifies **Iris flower species** — *Iris-setosa*, *Iris-versicolor*, and *Iris-virginica* — from four physical measurements. Built with Python and Scikit-learn, the project includes a modular training pipeline, model persistence, performance evaluation, and an interactive Streamlit web interface.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Dataset](#dataset)
- [Workflow](#workflow)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## Overview

The Iris dataset is a classic benchmark in supervised machine learning. This project demonstrates a complete classification workflow: data loading, preprocessing, model training, evaluation on held-out test data, artifact export, and real-time inference through a web application.

The solution is designed for clarity and reproducibility, making it suitable for learning fundamental ML concepts or serving as a foundation for more advanced classification projects.

---

## Key Features

- **Modular codebase** — Separated configuration, data loading, training, and inference logic
- **K-Nearest Neighbors classifier** — Simple, interpretable algorithm with strong performance on the Iris dataset
- **Stratified train/test split** — Maintains balanced class distribution across splits
- **Comprehensive evaluation** — Accuracy score, precision, recall, F1-score, and confusion matrix visualization
- **Model persistence** — Trained model saved with Joblib for reuse without retraining
- **Interactive web app** — Streamlit UI for entering measurements and receiving instant predictions
- **CLI training pipeline** — One-command training and evaluation via `main.py`

---

## Tech Stack

| Layer | Technology | Role |
|-------|------------|------|
| **Language** | Python 3.10+ | Application runtime and ML pipeline |
| **Machine Learning** | [Scikit-learn](https://scikit-learn.org/) | Classification model, metrics, and data splitting |
| **Data Processing** | [Pandas](https://pandas.pydata.org/) | CSV ingestion and feature matrix preparation |
| **Model Serialization** | [Joblib](https://joblib.readthedocs.io/) | Save and load trained model artifacts |
| **Visualization** | [Matplotlib](https://matplotlib.org/) | Confusion matrix plot generation |
| **Web Interface** | [Streamlit](https://streamlit.io/) | Browser-based prediction dashboard |
| **Dataset** | Iris.csv | 150 labeled samples across 3 species |

### Dependencies

```
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
joblib>=1.3.0
streamlit>=1.28.0
```

---

## Model Architecture

| Parameter | Value |
|-----------|-------|
| **Algorithm** | K-Nearest Neighbors (KNN) |
| **Neighbors (k)** | 3 |
| **Input Features** | 4 (sepal length, sepal width, petal length, petal width) |
| **Output Classes** | 3 (Iris-setosa, Iris-versicolor, Iris-virginica) |
| **Train/Test Split** | 80% / 20% |
| **Split Strategy** | Stratified |
| **Random State** | 42 |

---

## Project Structure

```
iris-flower-classification/
│
├── data/
│   └── Iris.csv                 # Source dataset
│
├── src/
│   ├── __init__.py
│   ├── config.py                # Paths, hyperparameters, and constants
│   ├── data_loader.py           # Dataset loading utilities
│   └── trainer.py               # Training, evaluation, and prediction
│
├── models/
│   └── iris_model.joblib        # Serialized model (generated)
│
├── outputs/
│   └── confusion_matrix.png     # Evaluation visualization (generated)
│
├── main.py                      # CLI entry point for training
├── app.py                       # Streamlit web application
├── requirements.txt             # Project dependencies
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10 or newer
- pip (Python package manager)
- Git (optional, for cloning the repository)

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/iris-flower-classification.git
cd iris-flower-classification
```

**2. Create a virtual environment (recommended)**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**

```bash
python -m pip install -r requirements.txt
```

> **Windows note:** If `pip install` fails with a launcher error, use `python -m pip install -r requirements.txt` instead. This occurs when an outdated Python installation is referenced by the `pip` shortcut.

---

## Usage

### Train the Model (CLI)

Run the full training and evaluation pipeline:

```bash
python main.py
```

**Expected output:**

- Dataset summary (150 samples, 4 features, 3 classes)
- Training and test set sizes
- Accuracy and classification report
- Saved model at `models/iris_model.joblib`
- Confusion matrix at `outputs/confusion_matrix.png`

### Launch the Web Application

Start the Streamlit prediction interface:

```bash
python -m streamlit run app.py
```

The application will open in your default browser. Enter flower measurements in centimeters and click **Predict species** to receive a classification result. If no trained model exists, the app will automatically train one on first launch.

---

## Model Performance

On the held-out test set (30 samples), the KNN classifier achieves:

| Metric | Score |
|--------|-------|
| **Accuracy** | 100% |
| **Precision (macro avg)** | 1.00 |
| **Recall (macro avg)** | 1.00 |
| **F1-Score (macro avg)** | 1.00 |

> Performance may vary slightly depending on the random seed and split configuration. The Iris dataset is well-separated, so high accuracy is expected with KNN.

---

## Dataset

The [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) contains 150 records with the following attributes:

| Feature | Type | Description |
|---------|------|-------------|
| `SepalLengthCm` | float | Sepal length in centimeters |
| `SepalWidthCm` | float | Sepal width in centimeters |
| `PetalLengthCm` | float | Petal length in centimeters |
| `PetalWidthCm` | float | Petal width in centimeters |
| `Species` | string | Target label (3 classes) |

**Target classes:**

- `Iris-setosa`
- `Iris-versicolor`
- `Iris-virginica`

---

## Workflow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  Load Data  │ ──▶ │  Train/Test  │ ──▶ │ Train Model │ ──▶ │   Evaluate   │
│  (Iris.csv) │     │    Split     │     │    (KNN)    │     │   & Export   │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                                                                      │
                                                                      ▼
                                                              ┌──────────────┐
                                                              │  Streamlit   │
                                                              │  Web App     │
                                                              └──────────────┘
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `pip` launcher error on Windows | Use `python -m pip install -r requirements.txt` |
| `streamlit` command not found | Use `python -m streamlit run app.py` |
| Model file missing | Run `python main.py` to train and save the model |
| Port already in use (Streamlit) | Run with a different port: `python -m streamlit run app.py --server.port 8502` |

---

## Future Enhancements

- [ ] Compare multiple algorithms (Logistic Regression, Random Forest, SVM)
- [ ] Hyperparameter tuning with cross-validation
- [ ] Feature importance and data exploration dashboard
- [ ] REST API endpoint for programmatic predictions
- [ ] Docker containerization for deployment
- [ ] CI/CD pipeline with automated tests

---

## Author

**Kalaiselvimurugan**

- GitHub:https://github.com/kalaiselvi2005975

---

<p align="center">
  Built with Python & Scikit-learn · A classic introduction to machine learning classification
</p>
