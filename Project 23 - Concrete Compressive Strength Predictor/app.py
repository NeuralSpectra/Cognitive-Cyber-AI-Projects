import streamlit as st
import pandas as pd
import pickle
import warnings

warnings.simplefilter("ignore")

with open('Scaler_X.pickle', 'rb') as f:
    scaler_X = pickle.load(f)

with open('Scaler_y.pickle', 'rb') as f:
    scaler_y = pickle.load(f)

with open('Finalized-Model.pickle', 'rb') as f:
    model = pickle.load(f)

st.title('Concrete Compressive Strength Predictor')

def user_input_features():
    cement = st.slider('Cement (kg/m³)', 0.0, 600.0, 200.0)
    fly_ash = st.slider('Fly Ash (kg/m³)', 0.0, 200.0, 50.0)
    water = st.slider('Water (kg/m³)', 0.0, 300.0, 150.0)
    superplasticizer = st.slider('Superplasticizer (kg/m³)', 0.0, 100.0, 10.0)
    age = st.slider('Age (days)', 1, 365, 28)

    return pd.DataFrame({
        'cement': [cement],
        'fly_ash': [fly_ash],
        'water': [water],
        'superplasticizer': [superplasticizer],
        'age': [age]
    })

input_df = user_input_features()

if st.button('Predict Concrete Compressive Strength'):
    scaled_input = scaler_X.transform(input_df)

    prediction_scaled = model.predict(scaled_input)

    predicted_strength = scaler_y.inverse_transform(
        prediction_scaled.reshape(-1, 1)
    )

    st.success(
        f'Estimated Concrete Compressive Strength: **{predicted_strength[0][0]:.2f} MPa**'
    )


