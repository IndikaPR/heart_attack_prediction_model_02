# app.py – FINAL: NO StreamlitAPIException + HIGH RISK
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
    st.success("High-risk patient loaded!")
    st.rerun()

# --- FORM (widgets use session_state) ---
with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 60, st.session_state.age, key="age_slider")
        gender = st.selectbox("Gender", ["Male", "Female"], index=0 if st.session_state.gender == "Male" else 1, key="gender_select")
        region = st.selectbox("Region", ["North", "South", "East", "West", "Central", "North-East"], 
                             index=["North", "South", "East", "West", "Central", "North-East"].index(st.session_state.region), key="region_select")
        urban_rural = st.selectbox("Urban/Rural", ["Urban", "Rural"], index=0 if st.session_state.urban_rural == "Urban" else 1, key="urban_select")
        ses = st.selectbox("SES", ["Low", "Middle", "High"], index=["Low", "Middle", "High"].index(st.session_state.ses), key="ses_select")
        smoking = st.selectbox("Smoking", ["
