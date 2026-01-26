import numpy as np
import streamlit as st
import cv2
from keras._tf_keras.keras.models import load_model
import warnings

warnings.simplefilter("ignore")

model = load_model('Finalized-Model.h5')
CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

st.title("Plant Disease Predictor")

plant_image = st.file_uploader("Choose An Image...", type=["jpg", "jpeg", "png"])
submit = st.button('Predict Plant Disease')

if submit:
    if plant_image is not None:
        try:
            file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            
            if opencv_image is None or opencv_image.size == 0:
                st.error("**Error: Could Not Load The Image. Please Upload A Valid Image.**")
            else:
                st.image(opencv_image, channels="BGR", caption="Uploaded Image")
                st.write(f"Original Size: {opencv_image.shape}")
                
                opencv_image = cv2.resize(opencv_image, (256, 256))
                opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
                opencv_image = opencv_image.astype(np.float32) / 255.0
                opencv_image = np.expand_dims(opencv_image, axis=0)
                
                Y_pred = model.predict(opencv_image, verbose=0)
                result_index = np.argmax(Y_pred)
                result = CLASS_NAMES[result_index]

                plant_name = result.split('-')[0]
                disease_name = result.split('-')[1].replace('_', ' ')
                st.success(f"**Prediction: {plant_name} leaf with {disease_name}**")
        
        except Exception as e:
            st.error(f"An Error Occurred: {str(e)}")
            st.write("Full Error Details:")
            st.exception(e)
    else:
        st.error("**Please Upload An Image Before Predicting.**")