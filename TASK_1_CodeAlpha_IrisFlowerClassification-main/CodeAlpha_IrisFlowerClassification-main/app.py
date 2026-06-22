"""Streamlit web app for Iris flower species prediction."""

from pathlib import Path

import streamlit as st

from src.config import MODEL_PATH
from src.trainer import load_model, predict, run_training

st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="centered")

st.title("Iris Flower Classification")
st.markdown(
    "Predict the species of an Iris flower from sepal and petal measurements "
    "using a K-Nearest Neighbors model trained with Scikit-learn."
)

if not MODEL_PATH.exists():
    st.warning("No trained model found. Training now...")
    run_training()

model = load_model(MODEL_PATH)

st.subheader("Enter flower measurements (cm)")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input("Sepal length", min_value=0.0, value=5.1, step=0.1)
    petal_length = st.number_input("Petal length", min_value=0.0, value=1.4, step=0.1)
with col2:
    sepal_width = st.number_input("Sepal width", min_value=0.0, value=3.5, step=0.1)
    petal_width = st.number_input("Petal width", min_value=0.0, value=0.2, step=0.1)

if st.button("Predict species", type="primary"):
    species = predict(model, sepal_length, sepal_width, petal_length, petal_width)
    st.success(f"Predicted species: **{species}**")

plot_path = Path(__file__).parent / "outputs" / "confusion_matrix.png"
if plot_path.exists():
    st.subheader("Model performance")
    st.image(str(plot_path), caption="Confusion matrix from last training run")
