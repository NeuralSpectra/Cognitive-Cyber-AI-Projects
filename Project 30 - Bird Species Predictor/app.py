import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pickle
import warnings

warnings.simplefilter("ignore")

st.set_page_config(page_title="Bird Species Classifier")

st.title("Bird Species Classifier")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("BirdClassifier_Balanced.keras")

@st.cache_data
def load_labels():
    with open("bird_labels.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
labels = load_labels()

def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize((150, 150))
    image = np.array(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            img = preprocess_image(image)
            preds = model.predict(img)[0]

            top_idx = np.argmax(preds)
            confidence = preds[top_idx] * 100
            species = labels[top_idx]

        st.success("Prediction Complete")
        st.markdown(f"### 🐤 **{species}**")
        st.markdown(f"**Confidence:** {confidence:.2f}%")

