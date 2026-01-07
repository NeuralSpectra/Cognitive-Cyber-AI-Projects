import streamlit as st
import pickle
import pandas as pd
import warnings

warnings.simplefilter("ignore")

USD_TO_INR = 83.38

with open("Finalized-Model.pickle", "rb") as f:
    model = pickle.load(f)

with open("Scaler_X.pickle", "rb") as f:
    scaler_X = pickle.load(f)

st.title("Petrol Price Predictor")

month = st.slider("Month", 1, 12, 1)
day = st.slider("Day", 1, 31, 1)
year = st.slider("Year", 2000, 2025, 2024)

if st.button("Predict Petrol Price"):
    input_data = pd.DataFrame(
        [[month, day, year]],
        columns=["Month", "Day", "Year"]
    )

    scaled_input = scaler_X.transform(input_data)
    predicted_price_usd = model.predict(scaled_input)[0]
    predicted_price_inr = predicted_price_usd * USD_TO_INR
    st.success(f"Predicted Petrol Price: **${predicted_price_inr:.2f}**")

