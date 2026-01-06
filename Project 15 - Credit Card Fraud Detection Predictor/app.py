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

FEATURES = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
]

def user_input_features():
    data = {}

    data['Time'] = st.slider('Time', 0.0, 172800.0, 50000.0)

    for col in FEATURES[1:-1]:
        data[col] = st.slider(col, -5.0, 5.0, 0.0)

    data['Amount'] = st.slider('Amount', 0.0, 5000.0, 100.0)

    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

scaled_input = scaler_X.transform(input_df)

if st.button('Fraud Detection'):
    prediction = model.predict(scaled_input)[0]

    result = '**🚨 Fraudulent Transaction**' if prediction == 1 else '**✅ Non-Fraudulent Transaction**'
    st.success(result)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(scaled_input)[0][1]
        st.info(f"**Fraud Probability: {prob:.2%}**")

