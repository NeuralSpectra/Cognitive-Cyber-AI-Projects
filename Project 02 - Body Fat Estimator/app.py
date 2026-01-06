import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="Body Fat Percentage Predictor", page_icon="🏋️", layout="centered")

@st.cache_resource
def load_model_and_scalers():
    with open('Finalized-Model.pickle', 'rb') as file:
        model = pickle.load(file)
    with open('Scaler_X.pickle', 'rb') as file:
        scaler_x = pickle.load(file)
    with open('Scaler_y.pickle', 'rb') as file:
        scaler_y = pickle.load(file)
    return model, scaler_x, scaler_y

try:
    model, scaler_x, scaler_y = load_model_and_scalers()
    st.success("✅ Model & Scalers Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Error Loading: {e}")
    st.stop()

st.title("🏋️ Body Fat Percentage Predictor")

if hasattr(scaler_x, 'feature_names_in_'):
    expected_features = list(scaler_x.feature_names_in_)
else:
    expected_features = ['Density', 'Abdomen', 'Chest', 'Hip', 'Weight']

input_values = {}
col1, col2 = st.columns(2)

feature_inputs = {
    'Density': {"min": 0.9, "max": 1.1, "value": 1.0, "step": 0.001, "format": "%.3f"},
    'Abdomen': {"min": 60.0, "max": 150.0, "value": 85.0, "step": 0.1},
    'Chest': {"min": 60.0, "max": 150.0, "value": 95.0, "step": 0.1},
    'Hip': {"min": 70.0, "max": 160.0, "value": 95.0, "step": 0.1},
    'Weight': {"min": 30.0, "max": 200.0, "value": 70.0, "step": 0.1},
}

for i, feature in enumerate(expected_features):
    col = col1 if i % 2 == 0 else col2
    with col:
        if feature in feature_inputs:
            params = feature_inputs[feature]
            if feature.lower() in ['density']:
                input_values[feature] = st.number_input(
                    f"{feature}", 
                    min_value=params["min"], 
                    max_value=params["max"], 
                    value=params["value"], 
                    step=params["step"],
                    format=params.get("format", "%f")
                )
            else:
                input_values[feature] = st.number_input(
                    f"{feature} (cm)" if feature.lower() not in ['weight', 'density'] else f"{feature} (kg)",
                    min_value=params["min"], 
                    max_value=params["max"], 
                    value=params["value"], 
                    step=params["step"]
                )

if st.button("🔍 Predict Body Fat Percentage", use_container_width=True):
    try:
        input_data = pd.DataFrame([[input_values[feature] for feature in expected_features]], 
                                 columns=expected_features)
        
        input_scaled = scaler_x.transform(input_data)
        prediction_scaled = model.predict(input_scaled)
        
        body_fat = float(scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0])
        
        if body_fat < 0 or body_fat > 50:
            st.error("🚨 **Invalid Prediction Detected**")
            st.warning(f"Predicted Body Fat: {body_fat:.2f}%")
            st.info("""
            **This negative value suggests:**
            - The model may be using wrong features
            - Training data might have issues
            - Scaling might be incorrect
            - Consider retraining with proper validation
            """)
        else:
            st.success(f"💪 Estimated Body Fat Percentage: **{body_fat:.2f}%**")
            
            # Interpretation
            if body_fat < 10:
                st.info("You Are Likely In An Athletic Range. Maintain With Proper Nutrition.")
            elif body_fat < 20:
                st.info("Healthy Range — Good Balance Between Muscle And Fat.")
            elif body_fat < 25:
                st.warning("Slightly Above Average — Consider Liight Fitness Improvements.")
            else:
                st.error("High body Fat Percentage. Consider Consulting A Fitness Professional.")
                
    except Exception as e:
        st.error(f"Prediction Failed: {e}")

