# **Project 29: Vehicle Count Predictor**

This project demonstrates **vehicle detection and counting** from images and video streams using **computer vision techniques**.

**It showcases how real-world CCTV-style vehicle counting systems work using classical OpenCV-based methods — without machine learning, deep learning, or neural networks.**

---

## **Libraries & Tools Used**

* **OpenCV (cv2)** – object detection and video processing
* **NumPy** – numerical operations
* **Pillow (PIL)** – image handling

⚠️ **No Machine Learning, Deep Learning, or Neural Networks are used in this project.**

---

## **Project Files**

Ensure the following files are placed in the **same directory**:

* **app.py**
* **Bus_front.xml** *(Haar Cascade for bus detection)*
* **cars.xml** *(Haar Cascade for car detection)*
* **cars.png** *(or any highway car image)*
* **bus.jpg** *(or any highway bus image)*
* **Cars.mp4** *(video stream for vehicle counting)*

---

## **How to Run the Application**

### **Step 1: Install Dependencies**

Make sure **Python 3.8+** is installed, then run:

   * **pip install opencv-python pillow numpy streamlit**

---

### **Step 2: Run the Streamlit App**

Open a terminal in the project directory and run:

**streamlit run app.py**

---

## **What the Application Does**

* Detects vehicles from:

  * Static images
  * Video streams
* Identifies:

  * Cars
  * Buses
* Displays bounding boxes around detected vehicles
* Demonstrates how vehicles are counted frame-by-frame

---

## **Important Notes**

* This project uses **Haar Cascade XML classifiers**
* Detection accuracy depends on:

  * Camera angle
  * Lighting conditions
  * Image/video resolution
* This is a **rule-based detection system**, not a learning-based one

---

## **Real-World Relevance**

This approach is commonly used in:

* Traffic monitoring systems
* Highway CCTV analytics
* Smart city surveillance
* Entry/exit vehicle counting systems
* Prototype traffic analysis tools

---

## **Note**

**This project is intentionally designed to demonstrate classical computer vision fundamentals before progressing into advanced AI-based traffic analysis systems.**