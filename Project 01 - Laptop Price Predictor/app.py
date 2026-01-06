import streamlit as st
import pickle
import pandas as pd
import numpy as np
import warnings

warnings.simplefilter("ignore")

@st.cache_resource
def load_artifacts():
    with open('Finalized-Model.pickle', 'rb') as file:
        model = pickle.load(file)
    with open('x_scaler.pickle', 'rb') as file:
        x_scaler = pickle.load(file)
    with open('y_scaler.pickle', 'rb') as file:
        y_scaler = pickle.load(file)
    return model, x_scaler, y_scaler

try:
    model, x_scaler, y_scaler = load_artifacts()
    scalers_loaded = True
except:
    st.error("Error Loading Model Or Scalers. Please Check If All Files Exist.")
    scalers_loaded = False

st.title("💻 Laptop Price Predictor")

col1, col2 = st.columns(2)

with col1:
    ram = st.number_input("RAM (GB)", min_value=2, max_value=64, value=8)
    weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=2.0)
    touchscreen = st.selectbox("TouchScreen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    ips = st.selectbox("IPS Display", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    ppi = st.number_input("PPI (Pixels Per Inch)", min_value=100, max_value=500, value=200)
    
    storage_type = st.selectbox("Storage Type", ["SSD", "HDD", "Both", "None"])
    if storage_type == "SSD":
        ssd, hdd = 1, 0
    elif storage_type == "HDD":
        ssd, hdd = 0, 1
    elif storage_type == "Both":
        ssd, hdd = 1, 1
    else:
        ssd, hdd = 0, 0

with col2:
    brands = ['Acer', 'Apple', 'Asus', 'Chuwi', 'Dell', 'Fujitsu', 'Google', 
              'HP', 'Huawei', 'LG', 'Lenovo', 'MSI', 'Mediacom', 'Microsoft', 
              'Razer', 'Samsung', 'Toshiba', 'Vero', 'Xiaomi']
    brand = st.selectbox("Brand", brands)
    
    laptop_types = ['2 in 1 Convertible', 'Gaming', 'Netbook', 'Notebook', 'Ultrabook', 'Workstation']
    laptop_type = st.selectbox("Laptop Type", laptop_types)
    
    os_types = ['Android', 'Chrome OS', 'Linux', 'Mac OS X', 'No OS', 
                'Windows 10', 'Windows 10 S', 'Windows 7', 'macOS']
    os_type = st.selectbox("Operating System", os_types)
    
    processor_brands = ['AMD Processor', 'Other Intel Processor', 'Intel']
    processor = st.selectbox("Processor Brand", processor_brands)
    
    gpu_brands = ['AMD', 'Intel', 'Nvidia']
    gpu = st.selectbox("GPU Brand", gpu_brands)

def create_feature_vector():
    features = {
        'Ram': ram,
        'Weight': weight,
        'TouchScreen': touchscreen,
        'IPS': ips,
        'PPI': ppi,
        'Acer': 0, 'Apple': 0, 'Asus': 0, 'Chuwi': 0, 'Dell': 0, 'Fujitsu': 0,
        'Google': 0, 'HP': 0, 'Huawei': 0, 'LG': 0, 'Lenovo': 0, 'MSI': 0,
        'Mediacom': 0, 'Microsoft': 0, 'Razer': 0, 'Samsung': 0, 'Toshiba': 0,
        'Vero': 0, 'Xiaomi': 0,
        '2 in 1 Convertible': 0, 'Gaming': 0, 'Netbook': 0, 'Notebook': 0,
        'Ultrabook': 0, 'Workstation': 0,
        'Android': 0, 'Chrome OS': 0, 'Linux': 0, 'Mac OS X': 0, 'No OS': 0,
        'Windows 10': 0, 'Windows 10 S': 0, 'Windows 7': 0, 'macOS': 0,
        'AMD Processor': 0, 'Other Intel Processor': 0,
        'HDD': hdd, 'SSD': ssd,
        'AMD': 0, 'Intel': 0, 'Nvidia': 0
    }
    
    features[brand] = 1
    features[laptop_type] = 1
    features[os_type] = 1
    features[processor] = 1
    features[gpu] = 1
    
    return pd.DataFrame([features])

def predict_price():
    try:
        input_df = create_feature_vector()
        input_scaled = x_scaler.transform(input_df)
        predicted_scaled = model.predict(input_scaled)        
        predicted_price = y_scaler.inverse_transform(predicted_scaled.reshape(-1, 1))
        
        return round(float(predicted_price[0][0]), 2)
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return None

if st.button("Predict Price") and scalers_loaded:
    with st.spinner("Calculating Price..."):
        price = predict_price()
        
        if price is not None:
            st.success(f"🏷️ Predicted Laptop Price: **₹{price:,.2f}**")
            
            st.subheader("📊 Selected Specifications:")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"- **Brand**: {brand}")
                st.write(f"- **RAM**: {ram} GB")
                st.write(f"- **Weight**: {weight} kg")
                st.write(f"- **TouchScreen**: {'Yes' if touchscreen else 'No'}")
            with col2:
                st.write(f"- **Storage**: {storage_type}")
                st.write(f"- **Type**: {laptop_type}")
                st.write(f"- **OS**: {os_type}")
                st.write(f"- **IPS Display**: {'Yes' if ips else 'No'}")

elif not scalers_loaded:
    st.warning("Please Ensure You Have Retrained The Model With Separate Scalers.")

