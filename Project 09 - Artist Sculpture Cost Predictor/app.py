import streamlit as st
import pandas as pd
import pickle
import warnings

warnings.simplefilter("ignore")


with open("Finalized-Model.pickle", "rb") as f:
    model = pickle.load(f)

st.title("Sculpture Cost Prediction App")
st.write("Fill the details below to predict the sculpture shipment cost.")

artist_reputation = st.number_input("Artist Reputation", min_value=0.0)
height = st.number_input("Height", min_value=0.0)
width = st.number_input("Width", min_value=0.0)
weight = st.number_input("Weight", min_value=0.0)
price_sculpture = st.number_input("Price Of Sculpture", min_value=0.0)
base_shipping = st.number_input("Base Shipping Price", min_value=0.0)

material = st.selectbox(
    "Material",
    ["Aluminium", "Brass", "Bronze", "Clay", "Marble", "Stone", "Wood"]
)

international = st.selectbox("International Shipping", ["No", "Yes"])
express = st.selectbox("Express Shipment", ["No", "Yes"])
installation = st.selectbox("Installation Included", ["No", "Yes"])
transport = st.selectbox("Transport Mode", ["Airways", "Roadways", "Waterways"])
fragile = st.selectbox("Fragile", ["No", "Yes"])
remote = st.selectbox("Remote Location", ["No", "Yes"])
customer_info = st.selectbox("Customer Information", ["Wealthy", "Working Class"])

material_columns = {
    "Material_Aluminium": 0,
    "Material_Brass": 0,
    "Material_Bronze": 0,
    "Material_Clay": 0,
    "Material_Marble": 0,
    "Material_Stone": 0,
    "Material_Wood": 0
}
material_columns[f"Material_{material}"] = 1

binary_fields = {
    "International_No": 1 if international == "No" else 0,
    "International_Yes": 1 if international == "Yes" else 0,
    "Express Shipment_No": 1 if express == "No" else 0,
    "Express Shipment_Yes": 1 if express == "Yes" else 0,
    "Installation Included_No": 1 if installation == "No" else 0,
    "Installation Included_Yes": 1 if installation == "Yes" else 0,
    "Transport_Airways": 1 if transport == "Airways" else 0,
    "Transport_Roadways": 1 if transport == "Roadways" else 0,
    "Transport_Waterways": 1 if transport == "Waterways" else 0,
    "Fragile_No": 1 if fragile == "No" else 0,
    "Fragile_Yes": 1 if fragile == "Yes" else 0,
    "Remote Location_No": 1 if remote == "No" else 0,
    "Remote Location_Yes": 1 if remote == "Yes" else 0,
    "Customer Information_Wealthy": 1 if customer_info == "Wealthy" else 0,
    "Customer Information_Working Class": 1 if customer_info == "Working Class" else 0
}

input_data = [
    artist_reputation,
    height,
    width,
    weight,
    price_sculpture,
    base_shipping,
] + list(material_columns.values()) + list(binary_fields.values())

input_df = pd.DataFrame([input_data])

if st.button("Predict Cost"):
    prediction = model.predict(input_df)[0]
    st.success(f"**Predicted Cost: {prediction:.2f}$**")
