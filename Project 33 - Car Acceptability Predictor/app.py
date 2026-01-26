import streamlit as st
import pickle
import pandas as pd
import warnings

warnings.simplefilter("ignore")

with open("Finalized-Model.pickle", "rb") as model_file:
    model = pickle.load(model_file)

with open("Scaler_X.pickle", "rb") as scaler_file:
    scaler_X = pickle.load(scaler_file)

feature_columns = [
    'doors', 'persons',
    'buying_high', 'buying_low', 'buying_med', 'buying_vhigh',
    'maint_high', 'maint_low', 'maint_med', 'maint_vhigh',
    'lug_boot_big', 'lug_boot_med', 'lug_boot_small',
    'safety_high', 'safety_low', 'safety_med'
]

mapping_class = {0: "unacc", 1: "acc", 2: "vgood", 3: "good"}

st.title('Car Acceptability Predictor')

doors = st.slider('Number Of Doors:', min_value=2, max_value=5)
persons = st.slider('Number Of Persons:', min_value=2, max_value=5)

buying = st.selectbox('Buying Price:', ['high', 'low', 'med', 'vhigh'])
maint = st.selectbox('Maintenance Cost:', ['high', 'low', 'med', 'vhigh'])
lug_boot = st.selectbox('Lug Boot Size:', ['big', 'med', 'small'])
safety = st.selectbox('Safety Rating:', ['high', 'low', 'med'])

def encode_features():
    data = {
        'doors': doors,
        'persons': persons,

        'buying_high': int(buying == 'high'),
        'buying_low': int(buying == 'low'),
        'buying_med': int(buying == 'med'),
        'buying_vhigh': int(buying == 'vhigh'),

        'maint_high': int(maint == 'high'),
        'maint_low': int(maint == 'low'),
        'maint_med': int(maint == 'med'),
        'maint_vhigh': int(maint == 'vhigh'),

        'lug_boot_big': int(lug_boot == 'big'),
        'lug_boot_med': int(lug_boot == 'med'),
        'lug_boot_small': int(lug_boot == 'small'),

        'safety_high': int(safety == 'high'),
        'safety_low': int(safety == 'low'),
        'safety_med': int(safety == 'med'),
    }
    return pd.DataFrame([data])[feature_columns]

if st.button('Predict Acceptability'):
    input_df = encode_features()
    input_scaled = scaler_X.transform(input_df)
    prediction = model.predict(input_scaled)
    label = mapping_class.get(prediction[0], "Unknown")

    st.success(f'The Predicted Class Is: **{label}**')

