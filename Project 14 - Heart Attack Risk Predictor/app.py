import streamlit as st
import numpy as np
import pickle
import warnings

warnings.simplefilter("ignore")

model = pickle.load(open('Finalized-Model.pickle', 'rb'))

def predict_heart_attack_risk(age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, caa):
    features = np.array([age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, caa]).reshape(1, -1)
    prediction = model.predict(features)
    return prediction

def main():
    st.title('Heart Attack Risk Predictor')

    st.markdown("---")

    age = st.slider('Age', 20, 100, 50)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
    trtbps = st.slider('Resting Blood Pressure (mm Hg)', 80, 200, 120)
    chol = st.slider('Cholesterol (mg/dl)', 100, 600, 200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['Yes', 'No'])
    restecg = st.selectbox('Resting ECG', ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'])
    thalachh = st.slider('Maximum Heart Rate Achieved', 60, 220, 150)
    exng = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])
    caa = st.slider('Number of Major Vessels (0-3)', 0, 3, 0)

    sex = 1 if sex == 'Male' else 0
    fbs = 1 if fbs == 'Yes' else 0
    exng = 1 if exng == 'Yes' else 0

    cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
    cp = cp_map[cp]

    restecg_map = {'Normal': 0, 'ST-T wave abnormality': 1, 'Left ventricular hypertrophy': 2}
    restecg = restecg_map[restecg]

    if st.button('Predict'):
        prediction = predict_heart_attack_risk(age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, caa)
        if prediction == 1:
            st.error("**The Person Is At High Risk Of A Heart Attack.**")
        else:
            st.success('**The Person Is Not At High Risk Of A Heart Attack.**')

if __name__ == '__main__':
    main()

