import streamlit as st
import warnings
import numpy as np
from PIL import Image
import cv2

warnings.simplefilter("ignore")

def add_text_watermark(image, text):
    image_np = np.array(image.convert("RGB"))
    h_image, w_image, _ = image_np.shape
    cv2.putText(image_np, text, (w_image - 95, h_image - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_4)
    return Image.fromarray(image_np, "RGB")

def add_image_watermark(base_image, watermark_image):

    base_image_np = np.array(base_image.convert("RGB"))
    watermark_image_np = np.array(watermark_image.convert("RGB"))

    h_base, w_base, _ = base_image_np.shape
    h_watermark, w_watermark, _ = watermark_image_np.shape

    center_y = int(h_base / 2)
    center_x = int(w_base / 2)
    top_y = center_y - int(h_watermark / 2)
    left_x = center_x - int(w_watermark / 2)
    bottom_y = top_y + h_watermark
    right_y = left_x + w_watermark

    roi = base_image_np[top_y: bottom_y, left_x : right_y]
    result = cv2.addWeighted(roi, 1, watermark_image_np, 1, 0)

    base_image_np[top_y : bottom_y, left_x: right_y] = result

    return Image.fromarray(base_image_np, "RGB")

st.title("Watermark Image Classifier")

option = st.selectbox("**Choose Watermark Type**", ["Text Watermark", "Image Watermark"])

if option == "Text Watermark":
    st.subheader("**Add Text Watermark**")
    uploaded_image = st.file_uploader("**Upload An Image:**", type=["png", "jpg", "jpeg"])
    text = st.text_input("**Enter Watermark Text**")

    if uploaded_image and text:
        image = Image.open(uploaded_image)
        watermarked_image = add_text_watermark(image, text)
        st.image(watermarked_image, caption="Watermarked Image", use_container_width=True)

elif option == "Image Watermark":
    st.subheader("**Add Image Watermark**")
    uploaded_image = st.file_uploader("**Upload Base Image**", type=["png", "jpg", "jpeg"])
    uploaded_watermark = st.file_uploader("**Upload Watermark Image**", type=["png", "jpg", "jpeg"])

    if uploaded_image and uploaded_watermark:
        base_image = Image.open(uploaded_image)
        watermark_image = Image.open(uploaded_watermark)
        watermarked_image =  add_image_watermark(base_image, watermark_image)
        st.image(watermarked_image, caption="Watermarked Image", use_container_width=True)


