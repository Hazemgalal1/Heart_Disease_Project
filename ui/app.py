import streamlit as st
import pandas as pd
import joblib
import os

MODEL_PATH = os.path.join("..", "models", "final_model.pkl")
if os.path.exists("models/final_model.pkl"):
    MODEL_PATH = "models/final_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    st.warning(f"Couldn't load model at {MODEL_PATH}: {e}")

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.write("ادخل بيانات المريض للتنبؤ باحتمالية الإصابة بأمراض القلب")

FEATURES = [
    "age", "sex", "trestbps", "chol", "thalach", "exang", "oldpeak",
    "cp_0.0","cp_1.0","cp_2.0","cp_3.0",
    "fbs_0.0","fbs_1.0",
    "restecg_0.0","restecg_1.0","restecg_2.0",
    "slope_0.0","slope_1.0","slope_2.0",
    "ca_0.0","ca_1.0","ca_2.0","ca_3.0",
    "thal_3.0","thal_6.0","thal_7.0"
]

inputs = {}
inputs['age'] = st.number_input('Age', 1, 120, 50)
inputs['sex'] = st.selectbox('Sex', [0,1], format_func=lambda x: "Male" if x==1 else "Female")
inputs['trestbps'] = st.number_input('Resting Blood Pressure', 50, 300, 120)
inputs['chol'] = st.number_input('Serum Cholesterol (mg/dl)', 50, 700, 200)
inputs['thalach'] = st.number_input('Max Heart Rate Achieved', 30, 300, 150)
inputs['exang'] = st.selectbox('Exercise Induced Angina (exang)', [0,1])
inputs['oldpeak'] = st.number_input('Oldpeak (ST depression)', 0.0, 10.0, 1.0, step=0.1)

cp = st.selectbox("Chest Pain Type (cp)", [0,1,2,3])
fbs = st.selectbox("Fasting Blood Sugar > 120 (fbs)", [0,1])
restecg = st.selectbox("Resting ECG (restecg)", [0,1,2])
slope = st.selectbox("Slope of ST segment (slope)", [0,1,2])
ca = st.selectbox("Number of major vessels (ca)", [0,1,2,3])
thal = st.selectbox("Thalassemia (thal)", [3,6,7])

row = {f: 0 for f in FEATURES}
row["age"] = inputs['age']
row["sex"] = inputs['sex']
row["trestbps"] = inputs['trestbps']
row["chol"] = inputs['chol']
row["thalach"] = inputs['thalach']
row["exang"] = inputs['exang']
row["oldpeak"] = inputs['oldpeak']
row[f"cp_{cp}.0"] = 1
row[f"fbs_{fbs}.0"] = 1
row[f"restecg_{restecg}.0"] = 1
row[f"slope_{slope}.0"] = 1
row[f"ca_{ca}.0"] = 1
row[f"thal_{thal}.0"] = 1

input_df = pd.DataFrame([row], columns=FEATURES)

st.subheader("🔎 البيانات المدخلة")
st.write(input_df)

if st.button("Predict"):
    if model is None:
        st.error("الموديل غير محمل. تأكد من وجود models/final_model.pkl")
    else:
        pred = model.predict(input_df)[0]
        prob = None
        try:
            proba = model.predict_proba(input_df)
            if proba.shape[1] > 2:
                best_class = proba.argmax(axis=1)[0]
                prob = proba.max(axis=1)[0]
                st.info(f"Predicted class: {best_class}  — Probability: {prob:.2f}")
            else:
                prob = proba[0][1]
        except Exception:
            prob = None

        if pred == 1:
            st.error("⚠️ الموديل يتوقع وجود خطر الإصابة بالقلب")
        else:
            st.success("✅ الموديل يتوقع عدم وجود خطر كبير")

        if prob is not None:
            st.caption(f"احتمالية (أقرب فئة): {prob:.2f}")
