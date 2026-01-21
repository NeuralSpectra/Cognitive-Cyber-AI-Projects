import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import os
import warnings

warnings.simplefilter("ignore")

def process_image_of_car(image_file):
    progress_bar = st.progress(0)
    
    image = Image.open(image_file)
    image = image.resize((450, 250))
    image_arr = np.array(image)

    gray = cv2.cvtColor(image_arr, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    dilated = cv2.dilate(blur, np.ones((3, 3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    cars_cascade_src = "cars.xml"
    cars_cascade = cv2.CascadeClassifier(cars_cascade_src)
    cars = cars_cascade.detectMultiScale(closing, 1.1, 1)

    cnt = 0
    for (x, y, w, h) in cars:
        cv2.rectangle(image_arr, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cnt += 1

    progress_bar.progress(100)
    st.image(image_arr, use_container_width=True)
    
    st.success(f"Image Processed Successfully!. {cnt} Cars Found!.")

def process_image_of_bus(image_file):
    progress_bar = st.progress(0)

    image = Image.open(image_file)
    image_arr = np.array(image)

    gray = cv2.cvtColor(image_arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    bus_cascade = cv2.CascadeClassifier("Bus_front.xml")

    if bus_cascade.empty():
        st.error("Bus cascade not loaded!")
        return

    buses = bus_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(60, 60)
    )

    cnt = 0
    for (x, y, w, h) in buses:
        cv2.rectangle(image_arr, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cnt += 1

    progress_bar.progress(100)
    st.image(image_arr, use_container_width=True)
    st.success(f"Image Processed Successfully!. {cnt} Buses Found!.")

def process_video(video_file):
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(video_file.getvalue())
    
    progress_bar = st.progress(0)
    
    cap = cv2.VideoCapture(temp_video_path)
    cars_cascade_src = "cars.xml"
    cars_cascade = cv2.CascadeClassifier(cars_cascade_src)

    frames = []
    cnt = 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    while True:
        ret, img = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cars = cars_cascade.detectMultiScale(gray, 1.1, 2)
        for (x, y, w, h) in cars:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cnt += 1
        frames.append(img)
        
        current_frame += 1
        progress = (current_frame / frame_count) * 100
        progress_bar.progress(int(progress))
        
        time.sleep(0.01)

    cap.release()

    out_path = "result.avi"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"DIVX"), 15, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(frame)
    out.release()

    os.remove(temp_video_path)

    progress_bar.progress(100)
    st.video(out_path)

    st.success(f"Video Processed Successfully!. {cnt} Cars Found!")

def main():
    st.title('Vehicle Count Detector')

    option = st.selectbox('Choose The Type Of Media',('Image', 'Video'))

    if option == 'Image':
        bus_or_car = st.selectbox("Choose Bus Or Car Image", ("Bus", "Car"))
        image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if bus_or_car == "Car" and image_file is not None:
            process_image_of_car(image_file)
        if bus_or_car == "Bus" and image_file is not None:
            process_image_of_bus(image_file)

    elif option == 'Video':
        video_file = st.file_uploader("Upload Video", type=["mp4"])
        if video_file is not None:
            process_video(video_file)

if __name__ == "__main__":
    main()
