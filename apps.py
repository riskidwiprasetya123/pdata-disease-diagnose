import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ==============================
# PAGE CONFIG# ==============================
st.set_page_config(
    page_title="Disease Diagnosis System",
    page_icon="🩺",
    layout="wide"
)

# ==============================
# LOAD MODEL & TOOLS
# ==============================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

le_gender = joblib.load("le_gender.pkl")
le_s1 = joblib.load("le_s1.pkl")
le_s2 = joblib.load("le_s2.pkl")
le_s3 = joblib.load("le_s3.pkl")
le_diag = joblib.load("le_diag.pkl")

treatment_map = joblib.load("treatment_map.pkl")
FEATURE_COLUMNS = joblib.load("features.pkl")

# ==============================
# HEADER
# ==============================
st.markdown("""
# 🩺 Disease Diagnosis System  
#### Supervised Learning-based Medical Prediction Application
---
""")

# ==============================
# SIDEBAR — INPUT
# ==============================
st.sidebar.header("👤 Patient Information")

patient_name = st.sidebar.text_input("Patient Name", placeholder="Enter patient name")
age = st.sidebar.number_input("Age", 1, 120, 25)
gender = st.sidebar.selectbox("Gender", le_gender.classes_)

st.sidebar.divider()
st.sidebar.subheader("🧾 Symptoms")
symptom1 = st.sidebar.selectbox("Symptom 1", le_s1.classes_)
symptom2 = st.sidebar.selectbox("Symptom 2", le_s2.classes_)
symptom3 = st.sidebar.selectbox("Symptom 3", le_s3.classes_)

st.sidebar.divider()
st.sidebar.subheader("❤️ Vital Signs")
heart_rate = st.sidebar.number_input("Heart Rate (bpm)", 40, 200, 75)
temperature = st.sidebar.number_input("Body Temperature (°C)", value=36.5)
oxygen = st.sidebar.number_input("Oxygen Saturation (%)", 70, 100, 98)

st.sidebar.divider()
st.sidebar.subheader("🩸 Blood Pressure")
systolic = st.sidebar.number_input("Systolic (mmHg)", 80, 250, 120)
diastolic = st.sidebar.number_input("Diastolic (mmHg)", 40, 150, 80)

predict_btn = st.sidebar.button("🔍 Run Diagnosis")

# ==============================
# MAIN LAYOUT
# ==============================
left_col, right_col = st.columns([1.3, 1])

with left_col:
    st.subheader("🧾 Patient Summary")

    check_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    receipt_text = f"""
------------------------------
      MEDICAL CHECK RECEIPT
------------------------------
Patient Name : {patient_name if patient_name else '-'}
Age          : {age}
Gender       : {gender}

Symptoms
- {symptom1}
- {symptom2}
- {symptom3}

Vital Signs
Heart Rate   : {heart_rate} bpm
Temperature  : {temperature} °C
Oxygen Sat   : {oxygen} %
Blood Press. : {systolic} / {diastolic} mmHg
------------------------------
Check Time   : {check_time}
------------------------------
"""

    st.code(receipt_text, language="text")


with right_col:
    st.subheader("🧠 Diagnosis Result")

    if predict_btn:
        if patient_name.strip() == "":
            st.error("⚠️ Please enter patient name before running diagnosis.")
        else:
            input_df = pd.DataFrame([{
                'Age': age,
                'Gender': le_gender.transform([gender])[0],
                'Symptom_1': le_s1.transform([symptom1])[0],
                'Symptom_2': le_s2.transform([symptom2])[0],
                'Symptom_3': le_s3.transform([symptom3])[0],
                'Heart_Rate_bpm': heart_rate,
                'Body_Temperature_C': temperature,
                'Oxygen_Saturation_%': oxygen,
                'Systolic': systolic,
                'Diastolic': diastolic
            }])[FEATURE_COLUMNS]

            input_scaled = scaler.transform(input_df)
            proba = model.predict_proba(input_scaled)[0]

            pred_encoded = np.argmax(proba)
            confidence = proba[pred_encoded] * 100

            diagnosis = le_diag.inverse_transform([pred_encoded])[0]
            treatment = treatment_map.get(
                pred_encoded,
                "Consult a medical professional."
            )

            st.success(f"🩺 **Diagnosis for {patient_name}:** {diagnosis}")
            st.metric("Confidence Level", f"{confidence:.2f}%")
            st.info(f"💊 **Recommended Treatment:** {treatment}")

    else:
        st.info("Please complete patient data and click **Run Diagnosis**")

# ==============================
# FOOTER
# ==============================
st.markdown("""
---
⚠️ **Disclaimer:**  
This application is developed for **educational and research purposes only**  
and does not replace professional medical diagnosis.
""")
