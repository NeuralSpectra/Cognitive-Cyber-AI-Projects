import streamlit as st
import pickle
import numpy as np
import warnings

warnings.simplefilter("ignore")

with open("Finalized-Model.pickle", "rb") as f:
    model = pickle.load(f)

with open("Scaler_X.pickle", "rb") as f:
    scaler_X = pickle.load(f)

with open("Scaler_y.pickle", "rb") as f:
    scaler_y = pickle.load(f)

if not hasattr(scaler_X, "feature_names_in_"):
    st.error(
        "Scaler was not fit on a DataFrame. "
        "Feature names are unavailable. "
        "Cannot safely extract min/max values."
    )
    st.stop()

def get_min_max(scaler, feature_name):
    idx = list(scaler.feature_names_in_).index(feature_name)
    return float(scaler.data_min_[idx]), float(scaler.data_max_[idx])

store_min, store_max = get_min_max(scaler_X, "Store")
dept_min, dept_max = get_min_max(scaler_X, "Dept")
temp_min, temp_max = get_min_max(scaler_X, "Temperature")
fuel_min, fuel_max = get_min_max(scaler_X, "Fuel_Price")

md1_min, md1_max = get_min_max(scaler_X, "MarkDown1")
md2_min, md2_max = get_min_max(scaler_X, "MarkDown2")
md3_min, md3_max = get_min_max(scaler_X, "MarkDown3")
md4_min, md4_max = get_min_max(scaler_X, "MarkDown4")
md5_min, md5_max = get_min_max(scaler_X, "MarkDown5")

cpi_min, cpi_max = get_min_max(scaler_X, "CPI")
unemp_min, unemp_max = get_min_max(scaler_X, "Unemployment")
size_min, size_max = get_min_max(scaler_X, "Size")
day_min, day_max = get_min_max(scaler_X, "Day")
month_min, month_max = get_min_max(scaler_X, "Month")
year_min, year_max = get_min_max(scaler_X, "Year")

st.title("Walmart Sales Forecasting")

store = st.slider("Store", int(store_min), int(store_max))
dept = st.slider("Department", int(dept_min), int(dept_max))
is_holiday = st.checkbox("Is Holiday")

temperature = st.slider("Temperature", temp_min, temp_max, step=0.1)
fuel_price = st.slider("Fuel Price", fuel_min, fuel_max, step=0.01)

markdown1 = st.slider("Markdown 1", md1_min, md1_max, step=0.01)
markdown2 = st.slider("Markdown 2", md2_min, md2_max, step=0.01)
markdown3 = st.slider("Markdown 3", md3_min, md3_max, step=0.01)
markdown4 = st.slider("Markdown 4", md4_min, md4_max, step=0.01)
markdown5 = st.slider("Markdown 5", md5_min, md5_max, step=0.01)

cpi = st.slider("CPI", cpi_min, cpi_max, step=0.01)
unemployment = st.slider("Unemployment", unemp_min, unemp_max, step=0.01)

type_map = {"A": 0, "B": 1, "C": 2}
store_type = st.selectbox("Store Type", list(type_map.keys()))

size = st.slider("Store Size", size_min, size_max)
day = st.slider("Day", int(day_min), int(day_max))
month = st.slider("Month", int(month_min), int(month_max))
year = st.slider("Year", int(year_min), int(year_max))

if st.button("Predict Walmart Sales"):

    input_data = np.array([[
        store,
        dept,
        int(is_holiday),
        temperature,
        fuel_price,
        markdown1,
        markdown2,
        markdown3,
        markdown4,
        markdown5,
        cpi,
        unemployment,
        type_map[store_type],
        size,
        day,
        month,
        year
    ]], dtype=float)

    input_scaled = scaler_X.transform(input_data)
    prediction_scaled = model.predict(input_scaled).reshape(-1, 1)
    prediction = scaler_y.inverse_transform(prediction_scaled)

    st.success(f"Predicted Weekly Sales: **${prediction[0][0]:,.2f}**")
