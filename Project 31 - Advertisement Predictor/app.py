import streamlit as st
import pandas as pd
import pickle
import warnings

warnings.simplefilter("ignore")

with open('Finalized-Model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('Scaler_X.pickle', 'rb') as f:
    scaler_X = pickle.load(f)

with open('LabelEncoders.pickle', 'rb') as f:
    label_encoders = pickle.load(f)

st.title('Ad Click Predictor')

daily_time_spent = st.number_input('Daily Time Spent on Site', min_value=0.0)
age = st.number_input('Age', min_value=0)
area_income = st.number_input('Area Income', min_value=0.0)
daily_internet_usage = st.number_input('Daily Internet Usage', min_value=0.0)

city = st.selectbox('City', label_encoders['City'].classes_)
country = st.selectbox('Country', label_encoders['Country'].classes_)
ad_topic_line = st.selectbox('Ad Topic Line', label_encoders['Ad Topic Line'].classes_)

gender_choice = st.selectbox('Gender', ['Male', 'Female'])
male = 0 if gender_choice == 'Male' else 1

year = st.slider('Year', min_value=1900, max_value=2100, value=2026, step=1)
month = st.slider('Month', min_value=1, max_value=12, value=1, step=1)
day = st.slider('Day', min_value=1, max_value=31, value=1, step=1)
hour = st.slider('Hour', min_value=0, max_value=23, value=0, step=1)
minute = st.slider('Minute', min_value=0, max_value=59, value=0, step=1)
second = st.slider('Second', min_value=0, max_value=59, value=0, step=1)

input_data = pd.DataFrame({
    'Daily Time Spent on Site': [daily_time_spent],
    'Age': [age],
    'Area Income': [area_income],
    'Daily Internet Usage': [daily_internet_usage],
    'City': [city],
    'Male': [male],
    'Country': [country],
    'Year': [year],
    'Month': [month],
    'Day': [day],
    'Hour': [hour],
    'Minute': [minute],
    'Second': [second],
    'Ad Topic Line': [ad_topic_line]
})

for col in ['City', 'Country', 'Ad Topic Line']:
    encoder = label_encoders[col]
    input_data[col] = encoder.transform(input_data[col].astype(str))

feature_cols = [
    'Daily Time Spent on Site', 'Age', 'Area Income',
    'Daily Internet Usage', 'City', 'Male', 'Country',
    'Year', 'Month', 'Day', 'Hour', 'Minute', 'Second',
    'Ad Topic Line'
]
input_data = input_data[feature_cols]

input_data_scaled = scaler_X.transform(input_data)

if st.button('Predict'):
    prediction = model.predict(input_data_scaled)
    probability = model.predict_proba(input_data_scaled)[0][1] if hasattr(model, "predict_proba") else None

    if prediction[0] == 1:
        st.success('**The Advertisement Was Clicked**')
    else:
        st.warning('**The Advertisement Was Not Clicked**')