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


feature_names = [
    'T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM'
]

column_descriptions = {
    'T': 'Average Annual Temperature',
    'TM': 'Annual Average Maximum Temperature',
    'Tm': 'Average Annual Minimum Temperature',
    'SLP': 'Sea Level Pressure',
    'H': 'Humidity',
    'VV': 'Average Visibility',
    'V': 'Annual Average Wind Speed',
    'VM': 'Maximum Wind Speed'
}

display_columns = [
    f'{col} ({column_descriptions[col]})' for col in feature_names
]

st.title('Air Quality Index Predictor')

st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 800px;
        padding: 1rem;
        margin: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def user_input_features():
    data = {}

    for i, col in enumerate(feature_names):
        min_val = float(scaler_X.data_min_[i])
        max_val = float(scaler_X.data_max_[i])
        mean_val = float((min_val + max_val) / 2)

        data[col] = st.slider(
            display_columns[i],
            min_val,
            max_val,
            mean_val
        )

    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

scaled_input = scaler_X.transform(input_df)

if st.button('Predict AQI'):
    prediction_scaled = model.predict(scaled_input)

    prediction_actual = scaler_y.inverse_transform(
        prediction_scaled.reshape(-1, 1)
    )

    rounded_AQI = round(float(prediction_actual[0][0]), 3)

    if rounded_AQI >= 100:
        msg = f"Air Quality Index Is **{rounded_AQI}**. The Air Quality Is **Very Bad**."
    elif rounded_AQI >= 50:
        msg = f"Air Quality Index Is **{rounded_AQI}**. The Air Quality Is **Moderate**."
    else:
        msg = f"Air Quality Index Is **{rounded_AQI}**. The Air Quality Is **Good**."

    st.success(msg)

