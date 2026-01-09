import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.simplefilter("ignore")

with open("Finalized-Model.pickle", 'rb') as f:
    model = pickle.load(f)

mapped_location = {'Portland':1, 'Cairns':2, 'Walpole':3, 'Dartmoor':4, 'MountGambier':5,
       'NorfolkIsland':6, 'Albany':7, 'Witchcliffe':8, 'CoffsHarbour':9, 'Sydney':10,
       'Darwin':11, 'MountGinini':12, 'NorahHead':13, 'Ballarat':14, 'GoldCoast':15,
       'SydneyAirport':16, 'Hobart':17, 'Watsonia':18, 'Newcastle':19, 'Wollongong':20,
       'Brisbane':21, 'Williamtown':22, 'Launceston':23, 'Adelaide':24, 'MelbourneAirport':25,
       'Perth':26, 'Sale':27, 'Melbourne':28, 'Canberra':29, 'Albury':30, 'Penrith':31,
       'Nuriootpa':32, 'BadgerysCreek':33, 'Tuggeranong':34, 'PerthAirport':35, 'Bendigo':36,
       'Richmond':37, 'WaggaWagga':38, 'Townsville':39, 'PearceRAAF':40, 'SalmonGums':41,
       'Moree':42, 'Cobar':43, 'Mildura':44, 'Katherine':45, 'AliceSprings':46, 'Nhil':47,
       'Woomera':48, 'Uluru':49}

rain_today_mapping = {"Yes": 1, "No": 0}

windgustdir = {'NNW':0, 'NW':1, 'WNW':2, 'N':3, 'W':4, 'WSW':5, 'NNE':6, 'S':7, 'SSW':8, 'SW':9, 'SSE':10,
       'NE':11, 'SE':12, 'ESE':13, 'ENE':14, 'E':15}
winddir9am = {'NNW':0, 'N':1, 'NW':2, 'NNE':3, 'WNW':4, 'W':5, 'WSW':6, 'SW':7, 'SSW':8, 'NE':9, 'S':10,
       'SSE':11, 'ENE':12, 'SE':13, 'ESE':14, 'E':15}
winddir3pm = {'NW':0, 'NNW':1, 'N':2, 'WNW':3, 'W':4, 'NNE':5, 'WSW':6, 'SSW':7, 'S':8, 'SW':9, 'SE':10,
       'NE':11, 'SSE':12, 'ENE':13, 'E':14, 'ESE':15}

st.title('Rain Predictor')

location = st.selectbox('Location', list(mapped_location.keys()))
mintemp = st.slider('MinTemp', -10.0, 50.0, 20.0)
maxtemp = st.slider('MaxTemp', -10.0, 50.0, 20.0)
rainfall = st.slider('Rainfall', 0.0, 500.0, 0.0)
evaporation = st.slider('Evaporation', 0.0, 200.0, 0.0)
sunshine = st.slider('Sunshine', 0.0, 15.0, 0.0)
windgustdir_input = st.selectbox('WindGustDir', list(windgustdir.keys()))
windgustspeed = st.slider('WindGustSpeed', 0.0, 200.0, 0.0)
winddir9am_input = st.selectbox('WindDir9am', list(winddir9am.keys()))
winddir3pm_input = st.selectbox('WindDir3pm', list(winddir3pm.keys()))
windspeed9am = st.slider('WindSpeed9am', 0.0, 100.0, 0.0)
windspeed3pm = st.slider('WindSpeed3pm', 0.0, 100.0, 0.0)
humidity9am = st.slider('Humidity9am', 0.0, 100.0, 50.0)
humidity3pm = st.slider('Humidity3pm', 0.0, 100.0, 50.0)
pressure9am = st.slider('Pressure9am', 950.0, 1050.0, 1010.0)
pressure3pm = st.slider('Pressure3pm', 950.0, 1050.0, 1010.0)
cloud9am = st.slider('Cloud9am', 0.0, 10.0, 0.0)
cloud3pm = st.slider('Cloud3pm', 0.0, 10.0, 0.0)
temp9am = st.slider('Temp9am', -10.0, 50.0, 20.0)
temp3pm = st.slider('Temp3pm', -10.0, 50.0, 20.0)
rain_today_input = st.selectbox("RainToday", list(rain_today_mapping.keys()))
rain_today_value = rain_today_mapping[rain_today_input]
year = st.slider('Year', 1900, 2100, 2024)
month = st.slider('Month', 1, 12, 1)
day = st.slider('Day', 1, 31, 1)

if st.button('Predict Rain Tomorrow'):
    input_data = pd.DataFrame({
        'Location': [mapped_location[location]],
        'MinTemp': [mintemp],
        'MaxTemp': [maxtemp],
        'Rainfall': [rainfall],
        'Evaporation': [evaporation],
        'Sunshine': [sunshine],
        'WindGustDir': [windgustdir[windgustdir_input]],
        'WindGustSpeed': [windgustspeed],
        'WindDir9am': [winddir9am[winddir9am_input]],
        'WindDir3pm': [winddir3pm[winddir3pm_input]],
        'WindSpeed9am': [windspeed9am],
        'WindSpeed3pm': [windspeed3pm],
        'Humidity9am': [humidity9am],
        'Humidity3pm': [humidity3pm],
        'Pressure9am': [pressure9am],
        'Pressure3pm': [pressure3pm],
        'Cloud9am': [cloud9am],
        'Cloud3pm': [cloud3pm],
        'Temp9am': [temp9am],
        'Temp3pm': [temp3pm],
        'RainToday': [rain_today_value],
        'Year': [year],
        'Month': [month],
        'Day': [day]
    })

    scaler_X = MinMaxScaler()
    input_data_scaled = scaler_X.fit_transform(input_data)

    prediction = model.predict(input_data_scaled)

    rain_tomorrow = '**It Might Rain Tomorrow**' if prediction[0] == 1 else '**It Might Not Rain Tomorrow**'

    st.success(rain_tomorrow)

