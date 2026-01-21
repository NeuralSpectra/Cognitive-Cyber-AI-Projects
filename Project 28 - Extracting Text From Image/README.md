# **Project 28: Extracting Text From Image**

This project demonstrates **text extraction (OCR)** from images using **Python** with the help of **Pillow, OpenCV, and Tesseract-OCR**.

**It marks the first project in this repository focused on extracting textual information from images — a foundational step toward document automation, OCR pipelines, and intelligent data extraction systems.**

---

## **Project Objective**

The purpose of this project is to:

* Demonstrate how **text can be extracted from any image**
* Introduce **Optical Character Recognition (OCR)** using **Tesseract**
* Combine **image preprocessing (OpenCV)** with **OCR (pytesseract)** for better accuracy
* Show how OCR is used in real-world applications such as:

  * Document digitization
  * ID card text extraction
  * Invoice and form processing

---

## **Libraries & Tools Used**

* **OpenCV (cv2)** – image preprocessing
* **Pillow (PIL)** – image handling
* **pytesseract** – Python wrapper for Tesseract OCR
* **Tesseract-OCR (Windows x64)** – OCR engine

---

## **How to Use the Application**

### **Step 1: Install Tesseract-OCR (Mandatory)**

⚠️ **This project requires Tesseract-OCR to be installed before running the app.**

1. Run the following file **first**:

   * **tesseract-ocr-w64-setup-5.4.0.20240606.exe**

   *(Windows x64 only)*

2. Install it in the **default location**:

   * **C:\Users\HP\AppData\Local\Programs\Tesseract-OCR\tesseract.exe**

   > Do **not** change the installation path.

---

### **Step 2: Install Python Dependencies**

Ensure **Python 3.8+** is installed, then run:

   * **pip install opencv-python pillow pytesseract streamlit**

---

### **Step 3: Project Structure**

Keep the following files in the same directory:

* **app.py**
* **pbIdS.png** *(or any image you want to test)*
* **tesseract-ocr-w64-setup-5.4.0.20240606.exe**

---

### **Step 4: Run the Application**

Open a terminal in the project directory and run:

**streamlit run app.py**

The app will:

* Load the image
* Preprocess it
* Extract and display text from the image

---

## **Important Notes**

* This project **does not involve machine learning training**
* OCR accuracy depends heavily on:

  * Image quality
  * Text clarity
  * Proper preprocessing
  
* Tesseract is an **external OCR engine**, not a Python-only package

---

## **Note**

**This is the first OCR-based project in the repository and intentionally focuses on clarity, setup correctness, and fundamentals before moving into advanced document intelligence systems.**