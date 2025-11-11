# app.py – FINAL: Load Patient + Accurate Prediction
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# --- Load model & tools ---
@st.cache_resource
def load_resources():
    model = load_model('heart_attack_model.h5')
    scaler = joblib.load('scaler.pkl')
    encoders = joblib.load('encoders.pkl')
    cols = joblib.load('columns.pkl')
    return model, scaler, encoders, cols

model, scaler, encoders, columns = load_resources()

# --- Initialize session state ---
if 'patient' not in st.session_state:
    st.session_state.patient = {}

# --- UI ---
st.title("Heart Attack Risk Predictor")
st.write("Enter patient details or **load high-risk example**")

# --- Load High-Risk Patient (24F) ---
if st.button("Load Real High-Risk Patient (24F)"):
    st.session_state.patient = {
        'age': 24, 'gender': 'Female', 'region': 'North', 'urban_rural': 'Urban',
        'ses': 'Low', 'smoking': 'Occasionally', 'alcohol': 'Occasionally', 'diet': 'Vegan',
        'activity': 'High', 'screen_time': 15, 'sleep': 3,
        'family_hx': 'Yes', 'diabetes': 'Yes', 'hypertension': 'No',
        'cholesterol': 256, 'bmi': 33.9, 'resting_hr': 86,
        'ecg': 'Normal', 'chest_pain': 'Typical', 'max_hr': 164,
        'angina': 'No', 'spo2': 92.7, 'triglycerides': 373,
        'systolic': 138, 'diastolic': 77
    }
    st.success("High-risk patient loaded!")
    st.rerun()

# --- Input Form (Use session_state or default) ---
col1, col2 = st.columns(2)

with col1:
    age = st.slider("**Age**", 18, 60, st.session_state.patient.get('age', 30))
    gender = st.selectbox("**Gender**", ["Male", "Female"], index=["Male", "Female"].index(st.session_state.patient.get('gender', 'Male')))
    region = st.selectbox("**Region**", ["North", "South", "East", "West", "Central", "North-East"], index=["North", "South", "East", "West", "Central", "North-East"].index(st.session_state.patient.get('region', 'East')))
    urban_rural = st.selectbox("**Urban/Rural**", ["Urban", "Rural"], index=["Urban", "Rural"].index(st.session_state.patient.get('urban_rural', 'Urban')))
    ses = st.selectbox("**SES**", ["Low", "Middle", "High"], index=["Low", "Middle", "High"].index(st.session_state.patient.get('ses', 'Middle')))
    smoking = st.selectbox("**Smoking**", ["Never", "Occasionally", "Regularly"], index=["Never", "Occasionally", "Regularly"].index(st.session_state.patient.get('smoking', 'Never')))
    alcohol = st.selectbox("**Alcohol**", ["Never", "Occasionally", "Regularly"], index=["Never", "Occasionally", "Regularly"].index(st.session_state.patient.get('alcohol', 'Never')))
    diet = st.selectbox("**Diet**", ["Vegetarian", "Non-Vegetarian", "Vegan"], index=["Vegetarian", "Non-Vegetarian", "Vegan"].index(st.session_state.patient.get('diet', 'Non-Vegetarian')))

with col2:
    activity = st.selectbox("**Activity**", ["Sedentary", "Moderate", "High"], index=["Sedentary", "Moderate", "High"].index(st.session_state.patient.get('activity', 'Moderate')))
    screen_time = st.slider("**Screen Time (hrs)**", 0, 16, st.session_state.patient.get('screen_time', 6))
    sleep = st.slider("**Sleep (hrs)**", 3, 12, st.session_state.patient.get('sleep', 7))
    family_hx = st.selectbox("**Family Hx**", ["Yes", "No"], index=["Yes", "No"].index(st.session_state.patient.get('family_hx', 'No')))
    diabetes = st.selectbox("**Diabetes**", ["Yes", "No"], index=["Yes", "No"].index(st.session_state.patient.get('diabetes', 'No')))
    hypertension = st.selectbox("**Hypertension**", ["Yes", "No"], index=["Yes", "No"].index(st.session_state.patient.get('hypertension', 'No')))
    cholesterol = st.slider("**Cholesterol**", 100, 400, st.session_state.patient.get('cholesterol', 180))
    bmi = st.slider("**BMI**", 15.0, 50.0, st.session_state.patient.get('bmi', 24.0), step=0.1)

st.markdown("---")
st.subheader("Clinical Data")
col3, col4 = st.columns(2)
with col3:
    resting_hr = st.slider("**Resting HR**", 50, 120, st.session_state.patient.get('resting_hr', 75))
    ecg = st.selectbox("**ECG**", ["Normal", "Abnormal"], index=["Normal", "Abnormal"].index(st.session_state.patient.get('ecg', 'Normal')))
    chest_pain = st.selectbox("**Chest Pain**", ["Typical", "Atypical", "Non-anginal", "Asymptomatic"], index=["Typical", "Atypical", "Non-anginal", "Asymptomatic"].index(st.session_state.patient.get('chest_pain', 'Non-anginal')))

with col4:
    max_hr = st.slider("**Max HR**", 80, 220, st.session_state.patient.get('max_hr', 150))
    angina = st.selectbox("**Exercise Angina**", ["Yes", "No"], index=["Yes", "No"].index(st.session_state.patient.get('angina', 'No')))
    spo2 = st.slider("**SpO2 (%)**", 85.0, 100.0, st.session_state.patient.get('spo2', 96.0), step=0.1)
    triglycerides = st.slider("**Triglycerides**", 50, 500, st.session_state.patient.get('triglycerides', 150))
    systolic = st.slider("**Systolic BP**", 90, 200, st.session_state.patient.get('systolic', 120))
    diastolic = st.slider("**Diastolic BP**", 60, 120, st.session_state.patient.get('diastolic', 80))

# --- Predict ---
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
        st.error("**HIGH RISK – URGENT CARDIOLOGY REFERRAL!**")
    else:
        st.success("Low risk – continue monitoring.")
