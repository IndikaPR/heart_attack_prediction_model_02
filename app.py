# app.py – FINAL VERSION (Streamlit 2025 Compatible)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

@st.cache_resource
def load():
    model = load_model('heart_attack_model.h5')
    scaler = joblib.load('scaler.pkl')
    encoders = joblib.load('encoders.pkl')
    cols = joblib.load('columns.pkl')
    return model, scaler, encoders, cols

model, scaler, encoders, columns = load()

st.title("Heart Attack Risk Predictor")
st.write("Enter **all patient details** – **sliders fixed & validated**")

# === INPUT FORM ===
col1, col2 = st.columns(2)

with col1:
    age = st.slider("**Age**", 18, 60, 30, help="18–60 years")
    gender = st.selectbox("**Gender**", ["Male", "Female"])
    region = st.selectbox("**Region**", ["North", "South", "East", "West", "Central", "North-East"])
    urban_rural = st.selectbox("**Urban/Rural**", ["Urban", "Rural"])
    ses = st.selectbox("**SES**", ["Low", "Middle", "High"])
    smoking = st.selectbox("**Smoking**", ["Never", "Occasionally", "Regularly"])
    alcohol = st.selectbox("**Alcohol**", ["Never", "Occasionally", "Regularly"])
    diet = st.selectbox("**Diet**", ["Vegetarian", "Non-Vegetarian", "Vegan"])

with col2:
    activity = st.selectbox("**Activity Level**", ["Sedentary", "Moderate", "High"])
    screen_time = st.slider("**Screen Time (hrs)**", 0, 16, 8)
    sleep = st.slider("**Sleep (hrs)**", 3, 12, 7)
    family_hx = st.selectbox("**Family History**", ["Yes", "No"])
    diabetes = st.selectbox("**Diabetes**", ["Yes", "No"])
    hypertension = st.selectbox("**Hypertension**", ["Yes", "No"])
    cholesterol = st.slider("**Cholesterol (mg/dL)**", 100, 400, 200)
    bmi = st.slider("**BMI**", 15.0, 50.0, 25.0, step=0.1)

st.markdown("---")
st.subheader("Clinical Data")
col3, col4 = st.columns(2)
with col3:
    resting_hr = st.slider("**Resting HR (bpm)**", 50, 120, 75)
    ecg = st.selectbox("**ECG**", ["Normal", "Abnormal"])
    chest_pain = st.selectbox("**Chest Pain**", ["Typical", "Atypical", "Non-anginal", "Asymptomatic"])

with col4:
    max_hr = st.slider("**Max HR Achieved**", 80, 220, 150)
    angina = st.selectbox("**Exercise Angina**", ["Yes", "No"])
    spo2 = st.slider("**SpO2 (%)**", 85.0, 100.0, 96.0, step=0.1)
    triglycerides = st.slider("**Triglycerides (mg/dL)**", 50, 500, 150)
    systolic = st.slider("**Systolic BP**", 90, 200, 120)
    diastolic = st.slider("**Diastolic BP**", 60, 120, 80)

# === LOAD HIGH-RISK PATIENT (24F) ===
if st.button("Load Real High-Risk Patient (24F)"):
    age, gender, region = 24, "Female", "North"
    urban_rural, ses = "Urban", "Low"
    smoking, alcohol, diet = "Occasionally", "Occasionally", "Vegan"
    activity, screen_time, sleep = "High", 15, 3
    family_hx, diabetes, hypertension = "Yes", "Yes", "No"
    cholesterol, bmi = 256, 33.9
    resting_hr, ecg, chest_pain = 86, "Normal", "Typical"
    max_hr, angina, spo2, triglycerides = 164, "No", 92.7, 373
    systolic, diastolic = 138, 77
    st.rerun()  # FIXED: Use st.rerun()

if st.button("Predict Risk", type="primary"):
    data = {
        'Age': age, 'Gender': gender, 'Region': region, 'Urban/Rural': urban_rural,
        'SES': ses, 'Smoking Status': smoking, 'Alcohol Consumption': alcohol,
        'Diet Type': diet, 'Physical Activity Level': activity,
        'Screen Time (hrs/day)': screen_time, 'Sleep Duration (hrs/day)': sleep,
        'Family History of Heart Disease': family_hx, 'Diabetes': diabetes,
        'Hypertension': hypertension, 'Cholesterol Levels (mg/dL)': cholesterol,
        'BMI (kg/m²)': bmi, 'Resting Heart Rate (bpm)': resting_hr,
        'ECG Results': ecg, 'Chest Pain Type': chest_pain,
        'Maximum Heart Rate Achieved': max_hr, 'Exercise Induced Angina': angina,
        'Blood Oxygen Levels (SpO2%)': spo2, 'Triglyceride Levels (mg/dL)': triglycerides,
        'Systolic_BP': systolic, 'Diastolic_BP': diastolic
    }

    df = pd.DataFrame([data])
    df = df.reindex(columns=columns, fill_value=0)

    for col in df.select_dtypes(include='object').columns:
        le = encoders.get(col)
        if le:
            df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    X_in = scaler.transform(df)
    prob = float(model.predict(X_in, verbose=0)[0][0])
    risk = "HIGH RISK" if prob > 0.5 else "LOW RISK"

    st.markdown(f"### **Risk Probability: {prob:.1%}**")
    st.markdown(f"### **Prediction: {risk}**")
    if prob > 0.5:
        st.error("**HIGH RISK – SEE CARDIOLOGIST NOW!**")
    else:
        st.success("Low risk – continue monitoring.")
