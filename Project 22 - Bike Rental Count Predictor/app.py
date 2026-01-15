import streamlit as st
import pandas as pd
import pickle
import warnings

warnings.simplefilter("ignore")

with open("Finalized-Model.pickle", "rb") as f:
    model = pickle.load(f)

with open("Scaler_X.pickle", "rb") as f:
    scaler_X = pickle.load(f)

with open("Scaler_y.pickle", "rb") as f:
    scaler_y = pickle.load(f)

st.title("🚲 Bike Rental Count Predictor")

def user_input_features():
    season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])
    year = st.slider("Year", 2011, 2012)
    month = st.slider("Month", 1, 12)
    holiday = st.selectbox("Holiday", [0, 1])
    weekday = st.slider("Weekday", 0, 6)
    workingday = st.selectbox("Working Day", [0, 1])
    weather_condition = st.slider("Weather Condition", 1, 4)
    temp = st.slider("Temperature (°C)", -10.0, 40.0, step=0.1)
    atemp = st.slider("Feels Like Temperature (°C)", -10.0, 40.0, step=0.1)
    humidity = st.slider("Humidity (%)", 0, 100)
    windspeed = st.slider("Windspeed (km/h)", 0.0, 70.0, step=0.1)
    casual = st.slider("Casual Rentals", 0, 2000)
    registered = st.slider("Registered Rentals", 0, 4000)

    season_mapping = {"Spring": 1, "Summer": 2, "Fall": 3, "Winter": 4}

    data = {
        "season": season_mapping[season],
        "year": year,
        "month": month,
        "holiday": holiday,
        "weekday": weekday,
        "workingday": workingday,
        "weather_condition": weather_condition,
        "temp": temp,
        "atemp": atemp,
        "humidity": humidity,
        "windspeed": windspeed,
        "casual": casual,
        "registered": registered,
        "Day": 0,
        "Month": 0,
        "Year": 0
    }

    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

def predict_rental_count(df):
    feature_order = [
        "season", "year", "month", "holiday", "weekday",
        "workingday", "weather_condition", "temp", "atemp",
        "humidity", "windspeed", "casual", "registered",
        "Day", "Month", "Year"
    ]

    df = df[feature_order]

    X_scaled = scaler_X.transform(df)
    y_scaled_pred = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_scaled_pred.reshape(-1, 1))

    return int(y_pred[0][0])

if st.button("Predict Rental Count"):
    prediction = predict_rental_count(input_df)
    st.success(f"Estimated Bike Rental Count: **{prediction}**")
