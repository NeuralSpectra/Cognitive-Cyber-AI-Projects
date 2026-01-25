import streamlit as st
import numpy as np
from keras.preprocessing.image import img_to_array
from PIL import Image
from keras.models import load_model
import warnings

warnings.simplefilter("ignore")

model = load_model("Finalized-Model.h5")

labels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

st.title("Intel Image Classifier")

uploaded_file = st.file_uploader("Choose An Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    image = image.resize((150, 150))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32') / 255.0

    if st.button('Classify Image'):
        prediction = model.predict(image)
        predicted_class = labels[np.argmax(prediction)]
        st.success(f'The Image Is Classified As: **{predicted_class}**')

