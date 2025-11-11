# STEP 13: Generate PERFECT app.py for Streamlit
app_code = '''
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

@st.cache_resource
def load_resources():
    model = load_model('heart_attack_model.h5')
    scaler = joblib.load('scaler.pkl')
    encoders = joblib.load('encoders.pkl')
    columns = joblib.load('columns.pkl')
    return model, scaler, encoders, columns

model, scaler, encoders, columns = load_resources()

# Default safe values (matches training data)
defaults = {
    'Age': 30,
    'Gender': 'Male',
    'Region': 'East',
    'Urban/Rural': 'Urban',
    'SES': 'Middle',
    'Smoking Status': 'Never',
    'Alcohol Consumption': 'Never',
    'Diet Type': 'Non-Vegetarian',
    'Physical Activity Level': 'Moderate',
    'Screen Time (hrs/day)': 6.0,
    'Sleep Duration (hrs/day)': 7.0,
    'Family History of Heart Disease': 'No',
    'Diabetes': 'No',
    'Hypertension': 'No',
    'Cholesterol Levels (mg/dL)': 180.0,
    'BMI (kg/m²)': 24.0,
    'Resting Heart Rate (bpm)': 75,
    'ECG Results': 'Normal',
    'Chest Pain Type': 'Non-anginal',
    'Maximum Heart Rate Achieved': 150,
    'Exercise Induced Angina': 'No',
    'Blood Oxygen Levels (SpO2%)': 96.0,
    'Triglyceride Levels (mg/dL)': 150.0,
    'Systolic_BP': 120,
    'Diastolic_BP': 80
}

st.title("Heart Attack Risk Predictor")
st.write("Enter **Age, Blood Pressure, and Gender**")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 60, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
with col2:
    systolic = st.slider("Systolic BP", 90, 200, 120)
    diastolic = st.slider("Diastolic BP", 60, 120, 80)

if st.button("Predict Risk", type="primary"):
    # Build input
    data = defaults.copy()
    data.update({
        'Age': age,
        'Gender': gender,
        'Systolic_BP': systolic,
        'Diastolic_BP': diastolic
    })
    
    df_input = pd.DataFrame([data])
    
    # Match exact training column order
    df_input = df_input.reindex(columns=columns, fill_value=0)
    
    # Encode categorical columns
    for col in df_input.select_dtypes(include='object').columns:
        le = encoders.get(col)
        if le:
            df_input[col] = df_input[col].map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    # Scale
    X_input = scaler.transform(df_input)
    
    # Predict
    prob = float(model.predict(X_input, verbose=0)[0][0])
    risk = "HIGH RISK" if prob > 0.5 else "LOW RISK"
    
    st.markdown(f"### Risk Probability: **{prob:.1%}**")
    st.markdown(f"### Final Prediction: **{risk}**")
    
    if prob > 0.5:
        st.error("IMMEDIATE MEDICAL ATTENTION RECOMMENDED!")
    else:
        st.success("Low risk. Maintain healthy lifestyle.")
'''

with open('app.py', 'w') as f:
    f.write(app_code)

files.download('app.py')
print("app.py generated and downloaded")