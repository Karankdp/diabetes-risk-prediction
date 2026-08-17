import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")

@st.cache_resource
def load_artifact():
    try:
        return joblib.load("diabetes_model.joblib")
    except Exception as e:
        st.error(f"Model loading error: {type(e).__name__}: {e}")
        st.stop()

artifact = load_artifact()
model = artifact["model"]
scaler = artifact["scaler"]
feature_columns = artifact["feature_columns"]
zero_as_missing_cols = artifact["zero_as_missing_cols"]
median_fill_values = artifact["median_fill_values"]
threshold = artifact.get("optimal_threshold", 0.5)

st.title("Diabetes Risk Predictor")
st.caption(f"Model: {artifact['model_name']}  |  Decision threshold: {threshold:.2f}")
st.write(
    "Enter a patient's diagnostic measurements to estimate diabetes risk. "
    "Leave a field at 0 if unknown -- it will be filled with the training "
    "set's median for that measurement."
)

with st.form("patient_form"):
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 0)
        glucose = st.number_input("Glucose", 0, 300, 120)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", 0, 200, 70)
        skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
    with col2:
        insulin = st.number_input("Insulin", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.4)
        age = st.number_input("Age", 1, 120, 35)

    submitted = st.form_submit_button("Predict")

if submitted:
    raw = pd.DataFrame([{
        "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness, "Insulin": insulin, "BMI": bmi,
        "DiabetesPedigreeFunction": dpf, "Age": age,
    }])
    raw = raw.reindex(columns=feature_columns, fill_value=np.nan)
    raw[zero_as_missing_cols] = raw[zero_as_missing_cols].replace(0, np.nan)
    for col, val in median_fill_values.items():
        if col in raw.columns:
            raw[col] = raw[col].fillna(val)

    scaled = scaler.transform(raw)
    prob = model.predict_proba(scaled)[0, 1]

    if prob >= threshold:
        tier, color = "HIGH risk -- recommend clinical follow-up", "red"
    elif prob >= threshold * 0.6:
        tier, color = "MODERATE risk -- monitor / lifestyle counseling", "orange"
    else:
        tier, color = "LOW risk", "green"

    st.metric("Predicted diabetes probability", f"{prob*100:.1f}%")
    st.markdown(f"**Risk tier:** :{color}[{tier}]")
    st.progress(min(float(prob), 1.0))
    st.caption(
        f"(Using tuned decision threshold {threshold:.2f}, not the default 0.5 -- "
        f"see Section 9c for why.)"
    )
