import streamlit as st
import pickle
import numpy as np
import warnings

warnings.simplefilter("ignore")

with open('Finalized-Model.pickle', 'rb') as model_file:
    model = pickle.load(model_file)

with open('Scaler_X.pickle', 'rb') as scaler_x_file:
    scaler_x = pickle.load(scaler_x_file)

with open('Scaler_y.pickle', 'rb') as scaler_y_file:
    scaler_y = pickle.load(scaler_y_file)

FEATURE_NAMES = ["Year", "Month"]

feature_min = dict(zip(FEATURE_NAMES, scaler_x.data_min_))
feature_max = dict(zip(FEATURE_NAMES, scaler_x.data_max_))

st.title("Time Series Forecasting")

year = st.slider("Select Year", int(feature_min["Year"]), int(feature_max["Year"]), int((feature_min["Year"] + feature_max["Year"]) / 2))
month = st.slider("Select Month", int(feature_min["Month"]), int(feature_max["Month"]), int((feature_min["Month"] + feature_max["Month"]) / 2))

if st.button("Predict Passengers"):
    input_data = np.array([[year, month]])
    input_data_scaled = scaler_x.transform(input_data)

    if hasattr(input_data_scaled, "to_numpy"):
        input_data_scaled = input_data_scaled.to_numpy()
    input_data_scaled = input_data_scaled.reshape((1, 1, 2))
    prediction_scaled = model.predict(input_data_scaled)
    prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))
    st.success(f"**Predicted Passengers: {int(prediction[0][0])}**")
