import streamlit as st
import pandas as pd
import pickle
import numpy as np
import warnings

warnings.simplefilter("ignore")

with open("Finalized-Model.pickle", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Car Selling Price Predictor", layout="centered") 
st.title("Car Selling Price Predictor")
st.write("*Enter Car Details Below To Estimate Its Selling Price.*")

year = st.slider("Year of Manufacture", min_value=1990, max_value=2025, value=2015, step=1)
km_driven = st.slider("Kilometers Driven", min_value=0, max_value=500000, value=50000, step=1000)
mileage = st.slider("Mileage (kmpl)", min_value=5.0, max_value=35.0, value=18.0, step=0.1)
engine = st.slider("Engine Capacity (CC)", min_value=500, max_value=6000, value=1200, step=50)
max_power = st.slider("Max Power (bhp)", min_value=20, max_value=800, value=80, step=1)
seats = st.slider("Number of Seats", min_value=2, max_value=10, value=5, step=1)
torque_rpm = st.slider("Torque RPM", min_value=500, max_value=10000, value=4000, step=100)

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner_type = st.selectbox("Owner Type",["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Driver Car"])

fuel_cols = {"CNG":0, "Diesel":0, "LPG":0, "Petrol":0}
fuel_cols[fuel_type] = 1

seller_cols = {"Dealer":0, "Individual":0, "Trustmark Dealer":0}
seller_cols[seller_type] = 1

transmission_cols = {"Automatic":0, "Manual":0}
transmission_cols[transmission] = 1

owner_cols = {"First Owner":0, "Second Owner":0, "Third Owner":0, "Fourth & Above Owner":0, "Test Driver Car":0}
owner_cols[owner_type] = 1

input_data = pd.DataFrame([{
    "year": year,
    "km_driven": km_driven,
    "mil_kmpl": mileage,
    "engine_cc": engine,
    "max_power_new": max_power,
    "seats": seats,
    "torque_rpm": torque_rpm,
    **transmission_cols,
    **seller_cols,
    **owner_cols,
    **fuel_cols
}])

feature_order = [
 'year', 'km_driven', 'seats', 'torque_rpm', 'mil_kmpl',
 'engine_cc', 'max_power_new', 'Automatic', 'Manual', 'Dealer',
 'Individual', 'Trustmark Dealer', 'First Owner', 'Fourth & Above Owner',
 'Second Owner', 'Test Driver Car', 'Third Owner', 'CNG', 'Diesel',
 'LPG', 'Petrol'
]

input_data = input_data[feature_order]

if st.button("Predict Selling Price 💰"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Selling Price: ₹ **{prediction:,.2f}**")

