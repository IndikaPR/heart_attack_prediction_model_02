# app.py – FINAL: NO SYNTAX ERROR + HIGH RISK ON LOAD
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

# --- Initialize session state ---
if 'initialized' not in st.session_state:
    st.session_state.update({
        'age': 30, 'gender': 'Male', 'region': 'East', 'urban_rural': 'Urban',
        'ses': 'Middle', 'smoking': 'Never', 'alcohol': 'Never', 'diet': 'Non-Vegetarian',
        'activity': 'Moderate', 'screen_time': 6, 'sleep': 7, 'family_hx': 'No',
        'diabetes': 'No', 'hypertension': 'No', 'cholesterol': 180, 'bmi': 24.0,
        'resting_hr': 75, 'ecg': 'Normal', 'chest_pain': 'Non-anginal',
        'max_hr': 150, 'angina': 'No', 'spo2': 96.0, 'triglycerides': 150,
        'systolic': 120, 'diastolic': 80, 'initialized': True
    })

st.title("Heart Attack Risk Predictor")
st.write("Click **Load High-Risk Patient** → See **HIGH RISK**")

# --- LOAD HIGH-RISK PATIENT (24F) ---
if st.button("Load Real High-Risk Patient (24F)"):
    st.session_state.update({
        'age': 24, 'gender': 'Female', 'region': 'North', 'urban_rural': 'Urban',
        'ses': 'Low', 'smoking': 'Occasionally', 'alcohol': 'Occasionally', 'diet': 'Vegan',
        'activity': 'High', 'screen_time': 15, 'sleep': 3, 'family_hx': 'Yes',
        'diabetes': 'Yes', 'hypertension': 'No', 'cholesterol': 256, 'bmi': 33.9,
        'resting_hr': 86, 'ecg': 'Normal', 'chest_pain': 'Typical', 'max_hr': 164,
        'angina': 'No', 'spo2': 92.7, 'triglycerides': 373, 'systolic': 138, 'diastolic': 77
    })
    st.success("High-risk patient loaded! Click 'Predict Risk'.")
    st.rerun()

# --- FORM ---
with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 60, st.session_state.age, key="age")
        gender = st.selectbox("Gender", ["Male", "Female"], 
                             index=0 if st.session_state.gender == "Male" else 1, key="gender")
        region = st.selectbox("Region", ["North", "South", "East", "West", "Central", "North-East"], 
                             index=["North", "South", "East", "West", "Central", "North-East"].index(st.session_state.region), key="region")
        urban_rural = st.selectbox("Urban/Rural", ["Urban", "Rural"], 
                                  index=0 if st.session_state.urban_rural == "Urban" else 1, key="urban")
        ses = st.selectbox("SES", ["Low", "Middle", "High"], 
                          index=["Low", "Middle", "High"].index(st.session_state.ses), key="ses")
        smoking = st.selectbox("Smoking", ["Never", "Occasionally", "Regularly"], 
                              index=["Never", "Occasionally", "Regularly"].index(st.session_state.smoking), key="smoking")
        alcohol = st.selectbox("Alcohol", ["Never", "Occasionally", "Regularly"], 
                              index=["Never", "Occasionally", "Regularly"].index(st.session_state.alcohol), key="alcohol")
        diet = st.selectbox("Diet", ["Vegetarian", "Non-Vegetarian", "Vegan"], 
                           index=["Vegetarian", "Non-Vegetarian", "Vegan"].index(st.session_state.diet), key="diet")

    with col2:
        activity = st.selectbox("Activity", ["Sedentary", "Moderate", "High"], 
                               index=["Sedentary", "Moderate", "High"].index(st.session_state.activity), key="activity")
        screen_time = st.slider("Screen Time (hrs)", 0, 16, st.session_state.screen_time, key="screen")
        sleep = st.slider("Sleep (hrs)", 3, 12, st.session_state.sleep, key="sleep")
        family_hx = st.selectbox("Family Hx", ["Yes", "No"], 
                                index=0 if st.session_state.family_hx == "Yes" else 1, key="family")
        diabetes = st.selectbox("Diabetes", ["Yes", "No"], 
                               index=0 if st.session_state.diabetes == "Yes" else 1, key="diabetes")
        hypertension = st.selectbox("Hypertension", ["Yes", "No"], 
                                   index=0 if st.session_state.hypertension == "Yes" else 1, key="hypertension")
        cholesterol = st.slider("Cholesterol", 100, 400, st.session_state.cholesterol, key="chol")
        bmi = st.slider("BMI", 15.0, 50.0, st.session_state.bmi, step=0.1, key="bmi")

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        resting_hr = st.slider("Resting HR", 50, 120, st.session_state.resting_hr, key="hr")
        ecg = st.selectbox("ECG", ["Normal", "Abnormal"], 
                          index=0 if st.session_state.ecg == "Normal" else 1, key="ecg")
        chest_pain = st.selectbox("Chest Pain", ["Typical", "Atypical", "Non-anginal", "Asymptomatic"], 
                                 index=["Typical", "Atypical", "Non-anginal", "Asymptomatic"].index(st.session_state.chest_pain), key="pain")

    with col4:
        max_hr = st.slider("Max HR", 80, 220, st.session_state.max_hr, key="maxhr")
        angina = st.selectbox("Angina", ["Yes", "No"], 
                             index=0 if st.session_state.angina == "Yes" else 1, key="angina")
        spo2 = st.slider("SpO2 (%)", 85.0, 100.0, st.session_state.spo2, step=0.1, key="spo2")
        triglycerides = st.slider("Triglycerides", 50, 500, st.session_state.triglycerides, key="trig")
        systolic = st.slider("Systolic BP", 90, 200, st.session_state.systolic, key="sys")
        diastolic = st.slider("Diastolic BP", 60, 120, st.session_state.diastolic, key="dia")

    submitted = st.form_submit_button("Predict Risk", type="primary")

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
