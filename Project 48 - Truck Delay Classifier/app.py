import streamlit as st
import pickle
import numpy as np
import warnings

warnings.simplefilter("ignore")

with open("Finalized-Model.pickle", "rb") as f:
    model = pickle.load(f)

with open("Scaler_X.pickle", "rb") as f:
    scaler_x = pickle.load(f)

st.title("Truck Delay / Accident Classifier")

FEATURE_NAMES = [
    "precip_x", "humidity_x", "visibility_x", "pressure_x",
    "precip_y", "humidity_y", "visibility_y", "pressure_y",
    "distance", "average_hours", "no_of_vehicles",
    "Day", "Month", "Year"
]

feature_min = dict(zip(FEATURE_NAMES, scaler_x.data_min_))
feature_max = dict(zip(FEATURE_NAMES, scaler_x.data_max_))

precip_x = st.slider("Precipitation (Source)", float(feature_min["precip_x"]), float(feature_max["precip_x"]), float((feature_min["precip_x"] + feature_max["precip_x"]) / 2))
humidity_x = st.slider("Humidity (Source)", float(feature_min["humidity_x"]), float(feature_max["humidity_x"]), float((feature_min["humidity_x"] + feature_max["humidity_x"]) / 2))
visibility_x = st.slider("Visibility (Source)", float(feature_min["visibility_x"]), float(feature_max["visibility_x"]), float((feature_min["visibility_x"] + feature_max["visibility_x"]) / 2))
pressure_x = st.slider("Pressure (Source)", float(feature_min["pressure_x"]), float(feature_max["pressure_x"]), float((feature_min["pressure_x"] + feature_max["pressure_x"]) / 2))
precip_y = st.slider("Precipitation (Destination)", float(feature_min["precip_y"]), float(feature_max["precip_y"]), float((feature_min["precip_y"] + feature_max["precip_y"]) / 2))
humidity_y = st.slider("Humidity (Destination)", float(feature_min["humidity_y"]), float(feature_max["humidity_y"]), float((feature_min["humidity_y"] + feature_max["humidity_y"]) / 2))
visibility_y = st.slider("Visibility (Destination)", float(feature_min["visibility_y"]), float(feature_max["visibility_y"]), float((feature_min["visibility_y"] + feature_max["visibility_y"]) / 2))
pressure_y = st.slider("Pressure (Destination)", float(feature_min["pressure_y"]), float(feature_max["pressure_y"]), float((feature_min["pressure_y"] + feature_max["pressure_y"]) / 2))
distance = st.slider("Distance (km)", float(feature_min["distance"]), float(feature_max["distance"]), float((feature_min["distance"] + feature_max["distance"]) / 2))
average_hours = st.slider("Average Travel Hours", float(feature_min["average_hours"]), float(feature_max["average_hours"]), float((feature_min["average_hours"] + feature_max["average_hours"]) / 2))
no_of_vehicles = st.slider("Number of Vehicles", int(feature_min["no_of_vehicles"]), int(feature_max["no_of_vehicles"]), int((feature_min["no_of_vehicles"] + feature_max["no_of_vehicles"]) / 2))

day = st.slider("Day", 1, 31, 15)
month = st.slider("Month", 1, 12, 6)
year = st.slider("Year", 2000, 2030, 2024)

if st.button("Predict Accident"):
    input_data = np.array([[
        precip_x, humidity_x, visibility_x, pressure_x,
        precip_y, humidity_y, visibility_y, pressure_y,
        distance, average_hours, no_of_vehicles,
        day, month, year
    ]])

    input_scaled = scaler_x.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 0:
        st.success("**✅ No Accident Predicted**")
    else:
        st.error("**⚠️ Accident Predicted**")
