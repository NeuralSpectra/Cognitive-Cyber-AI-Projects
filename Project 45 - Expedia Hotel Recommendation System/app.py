import streamlit as st
import pandas as pd
import pickle
import warnings

warnings.simplefilter("ignore")

with open('Finalized-Model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('Scaler_X.pickle', 'rb') as f:
    scaler_X = pickle.load(f)

with open('Scaler_y.pickle', 'rb') as f:
    scaler_y = pickle.load(f)

st.title("Hotel Cluster Prediction")

st.markdown("""
    <style>
    .centered {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    </style>
    <div class="centered">
""", unsafe_allow_html=True)

feature_mins = scaler_X.data_min_
feature_maxs = scaler_X.data_max_

site_name = st.slider('Site Name', float(feature_mins[0]), float(feature_maxs[0]), float(feature_mins[0]))
posa_continent = st.slider('POSA Continent', float(feature_mins[0]), float(feature_maxs[0]), float(feature_mins[0]))
user_location_country = st.slider('User Location Country', float(feature_mins[0]), float(feature_maxs[0]), float(feature_mins[0]))
user_location_region = st.slider('User Location Region', float(feature_mins[0]), float(feature_maxs[0]), float(feature_mins[0]))
user_location_city = st.slider('User Location City', float(feature_mins[0]), float(feature_maxs[0]), float(feature_mins[0]))
orig_destination_distance = st.slider('Origin Destination Distance', float(feature_mins[0]), float(feature_maxs[0]), float(feature_mins[0]))
user_id = st.slider('User ID', float(feature_mins[0]), float(feature_maxs[0]), float(feature_mins[0]))
is_mobile = int(st.checkbox("Is Mobile"))
is_package = int(st.checkbox("Pachage"))
channel = st.slider('Channel', float(feature_mins[0]), float(feature_maxs[0]), float(feature_mins[0]))
srch_adults_cnt = st.slider('Search Adults Count', float(feature_mins[0]), float(feature_maxs[0]), float(feature_mins[0]))
srch_children_cnt = st.slider('Search Children Count', float(feature_mins[0]), float(feature_maxs[0]), float(feature_mins[0]))
srch_rm_cnt = st.slider('Search Room Count', float(feature_mins[0]), float(feature_maxs[0]), float(feature_mins[0]))
srch_destination_id = st.slider('Search Destination ID', float(feature_mins[0]), float(feature_maxs[0]), float(feature_mins[0]))
srch_destination_type_id = st.slider('Search Destination Type ID', float(feature_mins[0]), float(feature_maxs[0]), float(feature_mins[0]))
is_booking = int(st.checkbox("Booking"))
cnt = st.slider('Count', float(feature_mins[0]), float(feature_maxs[0]), float(feature_mins[0]))
hotel_continent = st.slider('Hotel Continent', float(feature_mins[0]), float(feature_maxs[0]), float(feature_mins[0]))
hotel_country = st.slider('Hotel Country', float(feature_mins[0]), float(feature_maxs[0]), float(feature_mins[0]))
hotel_market = st.slider('Hotel Market', float(feature_mins[0]), float(feature_maxs[0]), float(feature_mins[0]))
stay_duration = st.slider('Stay Duration', float(feature_mins[0]), float(feature_maxs[0]), float(feature_mins[0]))
no_of_prior_days_booking = st.slider('Number of Prior Days Booking', float(feature_mins[0]), float(feature_maxs[0]), float(feature_mins[0]))
Cin_day = st.slider('Check-in Day', float(feature_mins[0]), float(feature_maxs[0]), float(feature_mins[0]))
Cin_month = st.slider('Check-in Month', float(feature_mins[0]), float(feature_maxs[0]), float(feature_mins[0]))
Cin_year = st.slider('Check-in Year', float(feature_mins[0]), float(feature_maxs[0]), float(feature_mins[0]))

input_data = pd.DataFrame([[
    site_name, posa_continent, user_location_country,
    user_location_region, user_location_city,
    orig_destination_distance, user_id, is_mobile,
    is_package, channel, srch_adults_cnt,
    srch_children_cnt, srch_rm_cnt,
    srch_destination_id, srch_destination_type_id,
    is_booking, cnt, hotel_continent,
    hotel_country, hotel_market,
    stay_duration, no_of_prior_days_booking,
    Cin_day, Cin_month, Cin_year
]], 

columns=[
    'site_name', 'posa_continent', 'user_location_country',
    'user_location_region', 'user_location_city',
    'orig_destination_distance', 'user_id', 'is_mobile',
    'is_package', 'channel', 'srch_adults_cnt',
    'srch_children_cnt', 'srch_rm_cnt',
    'srch_destination_id', 'srch_destination_type_id',
    'is_booking', 'cnt', 'hotel_continent',
    'hotel_country', 'hotel_market',
    'stay_duration', 'no_of_prior_days_booking',
    'Cin_day', 'Cin_month', 'Cin_year'
])

input_scaled = scaler_X.transform(input_data)

if st.button('Predict Hotel Cluster'):
    y_scaled_pred = model.predict(input_scaled)
    prediction = scaler_y.inverse_transform(y_scaled_pred.reshape(-1, 1))
    st.success(f'Predicted Hotel Cluster: **{int(prediction[0][0])}**')