import streamlit as st
import numpy as np
import pickle
import warnings
from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

warnings.simplefilter("ignore")

with open("Finalized-Model.pickle", "rb") as f:
    model = pickle.load(f)

with open("Scaler_X.pickle", "rb") as f:
    scaler_X = pickle.load(f)

with open("Scaler_y.pickle", "rb") as f:
    scaler_y = pickle.load(f)

with open("Route_LE.pickle", "rb") as f:
    LE = pickle.load(f)

route_options = LE.inverse_transform(np.arange(len(LE.classes_)))

FEATURE_COLUMNS = [
    'Route', 'Date', 'Month', 'stop',
    'Arrival_Hour', 'Arrival_Minute',
    'Departure_Hour', 'Departure_Minute',

    'Air Asia', 'Air India', 'GoAir', 'IndiGo',
    'Jet Airways', 'Jet Airways Business',
    'Multiple carriers', 'Multiple carriers Premium economy',
    'SpiceJet', 'Trujet', 'Vistara', 'Vistara Premium economy',

    '1 Long layover', '1 Short layover', '2 Long layover',
    'Business class', 'Change airports',
    'In-flight meal not included',
    'No check-in baggage included',
    'No info', 'Red-eye flight',

    'Banglore', 'Chennai', 'Delhi', 'Kolkata', 'Mumbai',

    'Destination_Banglore', 'Destination_Cochin',
    'Destination_Delhi', 'Destination_Hyderabad',
    'Destination_Kolkata', 'Destination_New Delhi'
]

st.title("Flight Price Predictor")

selected_route = st.selectbox("Select Route", route_options)
date = st.slider("Date", 1, 31)
month = st.slider("Month", 1, 12)
stops = st.slider("Number of Stops", 0, 5, 0)

arrival_hour = st.slider("Arrival Hour", 0, 23)
arrival_minute = st.slider("Arrival Minute", 0, 59)

departure_hour = st.slider("Departure Hour", 0, 23)
departure_minute = st.slider("Departure Minute", 0, 59)

airlines = [
    'Air Asia', 'Air India', 'GoAir', 'IndiGo', 'Jet Airways', 'Jet Airways Business', 'Multiple carriers', 'Multiple carriers Premium economy',
    'SpiceJet', 'Trujet', 'Vistara', 'Vistara Premium economy']
selected_airline = st.selectbox("Select Airline", airlines)

additional_info = [
    '1 Long layover', '1 Short layover', '2 Long layover', 'Business class', 'Change airports', 'In-flight meal not included',
    'No check-in baggage included', 'No info', 'Red-eye flight']
selected_info = st.selectbox("Select Additional Info", additional_info)

sources = ['Banglore', 'Chennai', 'Delhi', 'Kolkata', 'Mumbai']
selected_source = st.selectbox("Select Source", sources)

destinations = ['Banglore', 'Cochin', 'Delhi', 'Hyderabad', 'Kolkata', 'New Delhi']
selected_destination = st.selectbox("Select Destination", destinations)

route_encoded = LE.transform([selected_route])[0]

def predict_price():
    input_data = np.zeros((1, len(FEATURE_COLUMNS)))

    def set_value(col, value=1):
        if col in FEATURE_COLUMNS:
            input_data[0, FEATURE_COLUMNS.index(col)] = value

    set_value("Date", date)
    set_value("Month", month)
    set_value("stop", stops)

    set_value("Arrival_Hour", arrival_hour)
    set_value("Arrival_Minute", arrival_minute)
    set_value("Departure_Hour", departure_hour)
    set_value("Departure_Minute", departure_minute)

    set_value(selected_airline)
    set_value(selected_info)
    set_value(selected_source)
    set_value(f"Destination_{selected_destination}")

    scaled_input = scaler_X.transform(input_data)
    scaled_prediction = model.predict(scaled_input)

    final_price = scaler_y.inverse_transform(scaled_prediction.reshape(-1, 1))

    input_data[0, FEATURE_COLUMNS.index("Route")] = route_encoded

    return final_price[0][0]

if st.button("Predict Price"):
    price = predict_price()
    st.success(f"Estimated Flight Price: **₹{price:,.2f}**")

