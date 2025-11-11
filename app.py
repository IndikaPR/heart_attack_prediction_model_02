# app.py – FINAL: WORKING LOAD BUTTON + HIGH RISK
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# --- Load resources ---
@st.cache_resource
def load_resources():
    model = load_model('heart_attack_model.h5')
    scaler = joblib.load('scaler.pkl')
    encoders = joblib.load('encoders.pkl')
    cols = joblib.load('columns.pkl')
    return model, scaler, encoders, cols

model, scaler, encoders, columns = load_resources()

st.title("Heart Attack Risk Predictor")
st.write("Click **Load High-Risk Patient** → See **HIGH RISK**")

# --- HIDDEN FORM TO FORCE RELOAD ---
with st.form("patient_form"):
    # Default values
    age = st.slider("Age", 18, 60, 30, key="age")
    gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
    region = st.selectbox("Region", ["North", "South", "East", "West", "Central", "North-East"], key="region")
    urban_rural = st.selectbox("Urban/Rural", ["Urban", "Rural"], key="urban_rural")
    ses = st.selectbox("SES", ["Low", "Middle", "High"], key="ses")
    smoking = st.selectbox("Smoking", ["Never", "Occasionally", "Regularly"], key="smoking")
    alcohol = st.selectbox("Alcohol", ["Never", "Occasionally", "Regularly"], key="alcohol")
    diet = st.selectbox("Diet", ["Vegetarian", "Non-Vegetarian", "Vegan"], key="diet")
    activity = st.selectbox("Activity", ["Sedentary", "Moderate", "High"], key="activity")
    screen_time = st.slider("Screen Time (hrs)", 0, 16, 6, key="screen_time")
    sleep = st.slider("Sleep (hrs)", 3, 12, 7, key="sleep")
    family_hx = st.selectbox("Family Hx", ["Yes", "No"], key="family_hx")
    diabetes = st.selectbox("Diabetes", ["Yes", "No"], key="diabetes")
    hypertension = st.selectbox("Hypertension", ["Yes", "No"], key="hypertension")
    cholesterol = st.slider("Cholesterol", 100, 400, 180, key="cholesterol")
    bmi = st.slider("BMI", 15.0, 50.0, 24.0, step=0.1, key="bmi")
    resting_hr = st.slider("Resting HR", 50, 120, 75, key="resting_hr")
    ecg = st.selectbox("ECG", ["Normal", "Abnormal"], key="ecg")
    chest_pain = st.selectbox("Chest Pain", ["Typical", "Atypical", "Non-anginal", "Asymptomatic"], key="chest_pain")
    max_hr = st.slider("Max HR", 80, 220, 150, key="max_hr")
    angina = st.selectbox("Angina", ["Yes", "No"], key="angina")
    spo2 = st.slider("SpO2 (%)", 85.0, 100.0, 96.0, step=0.1, key="spo2")
    triglycerides = st.slider("Triglycerides", 50, 500, 150, key="triglycerides")
    systolic = st.slider("Systolic BP", 90, 200, 120, key="systolic")
    diastolic = st.slider("Diastolic BP", 60, 120, 80, key="diastolic")

    # Submit button inside form
    submitted = st.form_submit_button("Predict Risk")

# --- LOAD HIGH-RISK PATIENT ---
if st.button("Load Real High-Risk Patient (24F)"):
    st.session_state.update({
        'age': 24, 'gender': 'Female', 'region': 'North', 'urban_rural': 'Urban',
        'ses': 'Low', 'smoking': 'Occasionally', 'alcohol': 'Occasionally', 'diet': 'Vegan',
        'activity': 'High', 'screen_time': 15, 'sleep': 3, 'family_hx': 'Yes',
        'diabetes': 'Yes', 'hypertension': 'No', 'cholesterol': 256, 'bmi': 33.9,
        'resting_hr': 86, 'ecg': 'Normal', 'chest_pain': 'Typical', 'max_hr': 164,
        'angina': 'No', 'spo2': 92.7, 'triglycerides': 373, 'systolic': 138, 'diastolic': 77
    })
    st.success("High-risk patient loaded! Click 'Predict Risk' below.")
    st.rerun()

# --- PREDICT ---
if submitted:
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
