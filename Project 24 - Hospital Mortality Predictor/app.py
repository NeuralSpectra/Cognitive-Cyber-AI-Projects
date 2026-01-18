import streamlit as st
import pandas as pd
import pickle
import warnings

warnings.simplefilter("ignore")


FEATURE_COLUMNS = [
    'group', 'age', 'gendera', 'BMI', 'hypertensive',
    'atrialfibrillation', 'CHD with no MI', 'diabetes',
    'deficiencyanemias', 'depression', 'Hyperlipemia',
    'Renal failure', 'COPD', 'heart rate',
    'Systolic blood pressure', 'Diastolic blood pressure',
    'Respiratory rate', 'temperature', 'SP O2', 'Urine output',
    'hematocrit', 'RBC', 'MCH', 'MCHC', 'MCV', 'RDW',
    'Leucocyte', 'Platelets', 'Neutrophils', 'Basophils',
    'Lymphocyte', 'PT', 'INR', 'NT-proBNP',
    'Creatine kinase', 'Creatinine', 'Urea nitrogen',
    'glucose', 'Blood potassium', 'Blood sodium',
    'Blood calcium', 'Chloride', 'Anion gap',
    'Magnesium ion', 'PH', 'Bicarbonate',
    'Lactic acid', 'PCO2', 'EF'
]

with open("Finalized-Model.pickle", "rb") as f:
    model = pickle.load(f)

with open("Scaler_X.pickle", "rb") as f:
    scaler_X = pickle.load(f)

with open("Scaler_y.pickle", "rb") as f:
    scaler_y = pickle.load(f)

st.title("Hospital Mortality Predictor")

def user_input_features():
    gender = st.selectbox("Gender", ["Male", "Female"])

    data = {
        'group': st.slider("Group", 0, 4),
        'age': st.slider("Age", 0, 120),
        'gendera': 1 if gender == "Female" else 0,
        'BMI': st.slider("BMI", 10.0, 50.0, step=0.1),
        'hypertensive': st.selectbox("Hypertensive", [0, 1]),
        'atrialfibrillation': st.selectbox("Atrial Fibrillation", [0, 1]),
        'CHD with no MI': st.selectbox("CHD with no MI", [0, 1]),
        'diabetes': st.selectbox("Diabetes", [0, 1]),
        'deficiencyanemias': st.selectbox("Deficiency Anemias", [0, 1]),
        'depression': st.selectbox("Depression", [0, 1]),
        'Hyperlipemia': st.selectbox("Hyperlipemia", [0, 1]),
        'Renal failure': st.selectbox("Renal Failure", [0, 1]),
        'COPD': st.selectbox("COPD", [0, 1]),
        'heart rate': st.slider("Heart Rate", 30, 200),
        'Systolic blood pressure': st.slider("Systolic BP", 60, 200),
        'Diastolic blood pressure': st.slider("Diastolic BP", 30, 120),
        'Respiratory rate': st.slider("Respiratory Rate", 5, 50),
        'temperature': st.slider("Temperature", 30.0, 42.0, step=0.1),
        'SP O2': st.slider("SP O2", 70, 100),
        'Urine output': st.slider("Urine Output", 0, 5000),
        'hematocrit': st.slider("Hematocrit", 20, 60),
        'RBC': st.slider("RBC", 1.0, 6.0, step=0.1),
        'MCH': st.slider("MCH", 20, 40),
        'MCHC': st.slider("MCHC", 30, 40),
        'MCV': st.slider("MCV", 60, 100),
        'RDW': st.slider("RDW", 10, 20),
        'Leucocyte': st.slider("Leucocyte", 1.0, 15.0, step=0.1),
        'Platelets': st.slider("Platelets", 50, 500),
        'Neutrophils': st.slider("Neutrophils", 0, 100),
        'Basophils': st.slider("Basophils", 0, 5),
        'Lymphocyte': st.slider("Lymphocyte", 0, 100),
        'PT': st.slider("PT", 10.0, 20.0, step=0.1),
        'INR': st.slider("INR", 0.5, 5.0, step=0.01),
        'NT-proBNP': st.slider("NT-proBNP", 0, 50000),
        'Creatine kinase': st.slider("Creatine Kinase", 0, 10000),
        'Creatinine': st.slider("Creatinine", 0.1, 10.0, step=0.01),
        'Urea nitrogen': st.slider("Urea Nitrogen", 0, 200),
        'glucose': st.slider("Glucose", 40, 500),
        'Blood potassium': st.slider("Blood Potassium", 2.5, 7.0, step=0.1),
        'Blood sodium': st.slider("Blood Sodium", 100, 200),
        'Blood calcium': st.slider("Blood Calcium", 5.0, 15.0, step=0.1),
        'Chloride': st.slider("Chloride", 70, 120),
        'Anion gap': st.slider("Anion Gap", 5, 20),
        'Magnesium ion': st.slider("Magnesium Ion", 0.5, 5.0, step=0.01),
        'PH': st.slider("pH", 6.0, 8.0, step=0.01),
        'Bicarbonate': st.slider("Bicarbonate", 10, 40),
        'Lactic acid': st.slider("Lactic Acid", 0.5, 10.0, step=0.1),
        'PCO2': st.slider("PCO2", 20, 80),
        'EF': st.slider("EF", 10, 80)
    }

    df = pd.DataFrame(data, index=[0])

    # 🔒 ENFORCE FEATURE ORDER (CRITICAL)
    df = df[FEATURE_COLUMNS]

    return df

input_df = user_input_features()

if st.button("Predict Mortality"):
    scaled_input = scaler_X.transform(input_df)
    pred_scaled = model.predict(scaled_input)
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))

    outcome = "Death" if pred[0][0] > 0.5 else "Alive"
    st.success(f"Predicted Outcome: **{outcome}**")

