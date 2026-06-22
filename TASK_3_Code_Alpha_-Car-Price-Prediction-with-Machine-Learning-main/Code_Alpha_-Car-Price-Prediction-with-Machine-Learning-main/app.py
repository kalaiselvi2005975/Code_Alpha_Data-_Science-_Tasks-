from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "car_price_model.joblib"
REPORTS_DIR = BASE_DIR / "reports"


st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")
st.title("Car Price Prediction App")
st.caption("Advanced resale price estimator built with feature engineering and ensemble regression.")


def infer_brand(vehicle_name: str) -> str:
    text = str(vehicle_name).strip().lower()
    keyword_map = {
        "Maruti": ["ritz", "sx4", "ciaz", "wagon r", "wagon", "swift", "vitara", "ertiga", "dzire", "alto", "ignis", "baleno", "omni", "s cross", "800"],
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


def build_feature_frame(df: pd.DataFrame, current_year: int) -> pd.DataFrame:
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
    data["Vehicle_Age"] = (current_year - data["Year"]).clip(lower=0)
    data["Log_Driven_kms"] = np.log1p(data["Driven_kms"])
    data["Km_Per_Year"] = data["Driven_kms"] / data["Vehicle_Age"].replace(0, 1)
    data["Price_Age_Ratio"] = data["Present_Price"] / (data["Vehicle_Age"] + 1)
    data["Owner_Group"] = np.where(data["Owner"] <= 1, "Low", np.where(data["Owner"] == 2, "Medium", "High"))
    return data


def load_bundle():
    if not MODEL_PATH.exists():
        st.error("Trained model not found. Run `python train_model.py` first.")
        st.stop()
    return joblib.load(MODEL_PATH)


bundle = load_bundle()
model = bundle["model"]
metrics = bundle["holdout_metrics"]
top_features = bundle.get("top_features", [])


with st.sidebar:
    st.header("Vehicle Inputs")
    car_name = st.text_input("Vehicle / Model Name", value="city")
    year = st.number_input("Manufacturing Year", min_value=2000, max_value=bundle["current_year"], value=2018, step=1)
    present_price = st.number_input("Present Price (lakhs)", min_value=0.10, max_value=100.0, value=10.0, step=0.1)
    driven_kms = st.number_input("Driven Kilometers", min_value=0, max_value=500000, value=25000, step=500)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    selling_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox("Owner Count", [0, 1, 2, 3])
    predict_now = st.button("Predict Price", use_container_width=True)


col1, col2, col3, col4 = st.columns(4)
col1.metric("Best Model", metrics["best_model"])
col2.metric("R²", f"{metrics['r2']:.3f}")
col3.metric("RMSE", f"{metrics['rmse']:.3f}")
col4.metric("MAE", f"{metrics['mae']:.3f}")

st.divider()

input_df = pd.DataFrame(
    [
        {
            "Car_Name": car_name,
            "Year": int(year),
            "Present_Price": float(present_price),
            "Driven_kms": int(driven_kms),
            "Fuel_Type": fuel_type,
            "Selling_type": selling_type,
            "Transmission": transmission,
            "Owner": int(owner),
        }
    ]
)

left, right = st.columns([1.1, 0.9])

with left:
    st.subheader("Prediction")
    st.dataframe(input_df, use_container_width=True, hide_index=True)
    if predict_now:
        model_input = build_feature_frame(input_df, bundle["current_year"])
        prediction = float(model.predict(model_input)[0])
        st.success(f"Estimated resale price: ₹ {prediction:.2f} lakhs")
        st.info("The prediction is based on original dataset fields plus engineered features such as brand goodwill, vehicle age, driven-km intensity, and vehicle category.")
    else:
        st.write("Adjust the values in the sidebar and click `Predict Price`.")

with right:
    st.subheader("Top Drivers")
    if top_features:
        top_df = pd.DataFrame(top_features)
        st.dataframe(top_df, use_container_width=True, hide_index=True)
    else:
        st.write("Top feature list will appear after training.")


st.divider()
plot_col1, plot_col2 = st.columns(2)

with plot_col1:
    st.subheader("Actual vs Predicted")
    plot_path = REPORTS_DIR / "actual_vs_predicted.png"
    if plot_path.exists():
        st.image(str(plot_path), use_container_width=True)
    else:
        st.write("Plot not available yet.")

with plot_col2:
    st.subheader("Feature Importance")
    plot_path = REPORTS_DIR / "feature_importance.png"
    if plot_path.exists():
        st.image(str(plot_path), use_container_width=True)
    else:
        st.write("Plot not available yet.")


st.divider()
st.subheader("How to run")
st.code("python train_model.py\npython -m streamlit run app.py", language="bash")
