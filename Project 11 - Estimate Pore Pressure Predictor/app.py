import streamlit as st
import pandas as pd
import pickle
import warnings

warnings.simplefilter("ignore")

with open("Finalized-Model.pickle", "rb") as f:
    model = pickle.load(f)

with open("Scaler_X.pickle", "rb") as f:
    Scaler_X = pickle.load(f)

with open("Scaler_y.pickle", "rb") as f:
    Scaler_y = pickle.load(f)

st.title("Pore Pressure Predictor")

depth = st.slider("DEPTH", min_value=0.0, max_value=10000.0, value=1000.0, step=1.0)
gr = st.slider("GR", min_value=0.0, max_value=200.0, value=60.0, step=0.1)
rhob = st.slider("RHOB", min_value=1.5, max_value=3.0, value=2.3, step=0.01)
vsh = st.slider("Vsh", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
caliper = st.slider("Caliper", min_value=6.0, max_value=16.0, value=8.5, step=0.1)
porosity = st.slider("Porosity", min_value=0.0, max_value=0.4, value=0.15, step=0.01)
resistivity = st.slider("Resistivity", min_value=0.1, max_value=200.0, value=10.0, step=0.1)
stress = st.slider("Stress", min_value=0.0, max_value=500000.0, value=50.0, step=10.0)

if st.button("Predict Pore Pressure"):

    input_data = pd.DataFrame({
        'DEPTH': [depth],
        'GR': [gr],
        'RHOB': [rhob],
        'Vsh': [vsh],
        'Caliper': [caliper],
        'Porosity': [porosity],
        'Resistivity': [resistivity],
        'Stress': [stress]
    })

    input_scaled = Scaler_X.transform(input_data)
    predicted_scaled = model.predict(input_scaled)
    predicted_real = Scaler_y.inverse_transform(pd.DataFrame(predicted_scaled))

    result = float(predicted_real[0][0])

    st.success(f"**Estimated Pore Pressure: {result:.2f}**")

