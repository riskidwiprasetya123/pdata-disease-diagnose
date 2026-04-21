import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Sistem Diagnosis Penyakit",
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
# 🩺 Sistem Diagnosis Penyakit  
#### Aplikasi Prediksi Medis Berbasis Supervised Learning
---
""")

# ==============================
# SIDEBAR — INPUT
# ==============================
st.sidebar.header("👤 Informasi Pasien")

patient_name = st.sidebar.text_input("Nama Pasien", placeholder="Masukkan nama pasien")
age = st.sidebar.number_input("Usia", 1, 120, 25)
gender = st.sidebar.selectbox("Jenis Kelamin", le_gender.classes_)

st.sidebar.divider()
st.sidebar.subheader("🧾 Gejala")
symptom1 = st.sidebar.selectbox("Gejala 1", le_s1.classes_)
symptom2 = st.sidebar.selectbox("Gejala 2", le_s2.classes_)
symptom3 = st.sidebar.selectbox("Gejala 3", le_s3.classes_)

st.sidebar.divider()
st.sidebar.subheader("❤️ Tanda Vital")
heart_rate = st.sidebar.number_input("Denyut Jantung (bpm)", 40, 200, 75)
temperature = st.sidebar.number_input("Suhu Tubuh (°C)", value=36.5)
oxygen = st.sidebar.number_input("Saturasi Oksigen (%)", 70, 100, 98)

st.sidebar.divider()
st.sidebar.subheader("🩸 Tekanan Darah")
systolic = st.sidebar.number_input("Sistolik (mmHg)", 80, 250, 120)
diastolic = st.sidebar.number_input("Diastolik (mmHg)", 40, 150, 80)

predict_btn = st.sidebar.button("🔍 Jalankan Diagnosis")

# ==============================
# MAIN LAYOUT
# ==============================
left_col, right_col = st.columns([1.3, 1])

with left_col:
    st.subheader("🧾 Ringkasan Pasien")

    check_time = datetime.now().strftime("%d-%m-%Y %H:%M")

    receipt_text = f"""
------------------------------
      KWITANSI PEMERIKSAAN MEDIS
------------------------------
Nama Pasien   : {patient_name if patient_name else '-'}
Usia          : {age} tahun
Jenis Kelamin : {gender}

Gejala
- {symptom1}
- {symptom2}
- {symptom3}

Tanda Vital
Denyut Jantung: {heart_rate} bpm
Suhu Tubuh    : {temperature} °C
Oksigen       : {oxygen} %
Tekanan Darah : {systolic} / {diastolic} mmHg
------------------------------
Waktu Cek     : {check_time}
------------------------------
"""

    st.code(receipt_text, language="text")

with right_col:
    st.subheader("🧠 Hasil Diagnosis")

    if predict_btn:
        if patient_name.strip() == "":
            st.error("⚠️ Masukkan nama pasien sebelum menjalankan diagnosis.")
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
                "Konsultasikan dengan dokter profesional."
            )

            st.success(f"🩺 **Diagnosis untuk {patient_name}:** {diagnosis}")
            st.metric("Tingkat Keyakinan", f"{confidence:.2f}%")
            st.info(f"💊 **Pengobatan yang Direkomendasikan:** {treatment}")

    else:
        st.info("Lengkapi data pasien dan klik **Jalankan Diagnosis**")

# ==============================
# FOOTER
# ==============================
st.markdown("""
---
⚠️ **Disclaimer:**  
Aplikasi ini dikembangkan untuk **tujuan pendidikan dan penelitian saja**  
dan tidak menggantikan diagnosis medis profesional.
""")
