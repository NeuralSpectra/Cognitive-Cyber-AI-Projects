import streamlit as st
import pandas as pd
import pickle
import warnings

warnings.simplefilter("ignore")

with open("Finalized-Model.pickle", "rb") as f:
    model = pickle.load(f)

with open("Scaler_X.pickle", "rb") as f:
    X_scaler = pickle.load(f)

with open("Scaler_y.pickle", "rb") as f:
    y_scaler = pickle.load(f)

business_code_map = {"U001": 0, "CA02": 1, "U013": 2, "U002": 3, "U005": 4, "U007": 5}

customer_payment_term_map = {
    'NAH4': 0, 'NAD1': 1, 'NAA8': 2, 'CA10': 3, 'NAC6': 4, 'NAM2': 5,
    'NAAX': 6, 'NAVE': 7, 'NAG2': 8, 'NABG': 9, 'NAM4': 10, 'NA10': 11,
    'NAU5': 12, 'NA32': 13, 'NAD5': 14, 'NAWP': 15, 'NAGD': 16,
    'NAVR': 17, 'CA30': 18, 'NAM1': 19, 'NAAW': 20, 'NAVF': 21,
    'NAD4': 22, 'NAUZ': 23, 'NA3F': 24, 'NAX2': 25, 'NAVQ': 26,
    'NATM': 27, 'CAB1': 28, 'NA84': 29, 'NAWM': 30, 'NACB': 31,
    'NACG': 32, 'NA38': 33, 'NAWN': 34, 'C106': 35, 'NAWU': 36,
    'NAB1': 37, 'NA3B': 38, 'NA9X': 39, 'NAVD': 40, 'NAVM': 41,
    'NACE': 42, 'NA25': 43, 'NAUP': 44, 'NAM3': 45, 'NACH': 46,
    'CAX2': 47, 'NATV': 48, 'NAVL': 49, 'NATZ': 50, 'C129': 51,
    'BR56': 52, 'NA31': 53, 'NATW': 54, 'B052': 55, 'NAV2': 56,
    'NATX': 57, 'NAUY': 58, 'NA8Q': 59, 'NATJ': 60, 'BR12': 61,
    'NATU': 62, '90M7': 63, 'NAV9': 64, 'NATK': 65, 'CA60': 66,
    'NATL': 67, 'NAD8': 68, 'NAUW': 69, 'NAVC': 70, 'NABD': 71,
    'NATH': 72, 'MC15': 73
}

st.title("Working Capital Optimization")

business_code = st.selectbox("Business Code", business_code_map.keys())
business_year = st.slider("Business Year", 2000, 2024, 2024)
payterm = st.slider("Payterm", 1, 100, 30)
total_open_amount = st.slider("Total Open Amount", 1, 100000, 50000)
total_open_amount_usd = st.slider("Total Open Amount (USD)", 1, 100000, 50000)
customer_payment_terms = st.selectbox("Customer Payment Terms", customer_payment_term_map.keys())
is_open = st.selectbox("Is Open", ["Yes", "No"])
dunnlevel = st.slider("DUNNLEVEL", 2010, 2024, 2024)

posting_day = st.slider("Posting Day", 1, 31, 1)
posting_month = st.slider("Posting Month", 1, 12, 1)
posting_year = st.slider("Posting Year", 2000, 2024, 2024)
due_day = st.slider("Due Day", 1, 31, 1)
due_month = st.slider("Due Month", 1, 12, 1)
due_year = st.slider("Due Year", 2000, 2024, 2024)
baseline_day = st.slider("Baseline Day", 1, 31, 1)
baseline_month = st.slider("Baseline Month", 1, 12, 1)
baseline_year = st.slider("Baseline Year", 2000, 2024, 2024)

region = st.selectbox("Region", ["MIDWEST", "NORTHEAST", "SOUTHEAST", "SOUTHWEST", "WEST"])
invoice_currency = st.selectbox("Invoice Currency", ["USD", "CAD"])

input_data = pd.DataFrame({
    'Business Code': [business_code_map[business_code]],
    'Business Year': [business_year],
    'Payterm': [payterm],
    'Total Open Amount': [total_open_amount],
    'USD_CURRENNCY': [0],
    'Total Open Amount_USD': [total_open_amount_usd],
    'Customer Payment Terms': [customer_payment_term_map[customer_payment_terms]],
    'Is Open': [1 if is_open == "Yes" else 0],
    'DUNNLEVEL': [dunnlevel],
    'Posting_Day': [posting_day],
    'Posting_Month': [posting_month],
    'Posting_Year': [posting_year],
    'Due_Day': [due_day],
    'Due_Month': [due_month],
    'Due_Year': [due_year],
    'Baseline_Day': [baseline_day],
    'Baseline_Month': [baseline_month],
    'Baseline_Year': [baseline_year],
    'Invoice Currency_CAD': [1 if invoice_currency == "CAD" else 0],
    'Invoice Currency_USD': [1 if invoice_currency == "USD" else 0],
    'MIDWEST': [1 if region == "MIDWEST" else 0],
    'NORTHEAST': [1 if region == "NORTHEAST" else 0],
    'SOUTHEAST': [1 if region == "SOUTHEAST" else 0],
    'SOUTHWEST': [1 if region == "SOUTHWEST" else 0],
    'WEST': [1 if region == "WEST" else 0]
})

if st.button("Predict Credit Limit"):
    input_scaled = X_scaler.transform(input_data)
    prediction_scaled = model.predict(input_scaled)
    prediction = y_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))

    st.success(f"Predicted Credit Limit: **{prediction[0][0]:,.2f}**")
