import streamlit as st
from skimage.metrics import structural_similarity as ssim
import cv2
from PIL import Image
import numpy as np
import warnings

warnings.simplefilter("ignore")

def load_image(image_file):
    img = Image.open(image_file)
    return img

def calculate_similarity(original_img, uploaded_img):
    original_img = original_img.resize((250, 160))
    uploaded_img = uploaded_img.resize((250, 160))

    original_img = np.array(original_img)
    uploaded_img = np.array(uploaded_img)

    original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    uploaded_gray = cv2.cvtColor(uploaded_img, cv2.COLOR_BGR2GRAY)

    score, diff = ssim(original_gray, uploaded_gray, full=True)
    return score * 100  

st.title("PAN Card Detection Predictor")

original_image_path = "Images\\pan-card.jpg"
original_image = load_image(original_image_path)
st.image(original_image, caption='Original PAN Card', use_container_width=True)

uploaded_file = st.file_uploader("Upload a PAN Card Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    uploaded_image = load_image(uploaded_file)
    st.image(uploaded_image, caption='Uploaded PAN Card', use_container_width=True)

    similarity_score = calculate_similarity(original_image, uploaded_image)

    st.success(f"Similarity Score: **{similarity_score:.2f}%**")

