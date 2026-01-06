import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings

warnings.simplefilter("ignore")


with open('Finalized-Model.pickle', 'rb') as file:
    model = pickle.load(file)


def predict_cost(age, sex, bmi, children, smoker, region):

    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'northeast': [1 if region == 'northeast' else 0],
        'northwest': [1 if region == 'northwest' else 0],
        'southeast': [1 if region == 'southeast' else 0],
        'southwest': [1 if region == 'southwest' else 0]
    })


    input_data['sex'] = input_data['sex'].apply(lambda x: 1 if x == 'male' else 0)
    input_data['smoker'] = input_data['smoker'].apply(lambda x: 1 if x == 'yes' else 0)


    prediction = model.predict(input_data)
    return prediction[0]


st.title("Medical Cost Predictor")


age = st.number_input("Age", min_value=0, max_value=120, value=25)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=20, value=0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])


if st.button("Predict"):
    result = predict_cost(age, sex, bmi, children, smoker, region)
    st.success(f'**The Estimated Medical Cost Is: ₹{result:.2f}**')


