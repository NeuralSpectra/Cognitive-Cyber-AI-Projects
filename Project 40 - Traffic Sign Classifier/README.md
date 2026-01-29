# **Project 40: Traffic Sign Classifier**

This project classifies **traffic sign images** using a **Convolutional Neural Network (CNN)** trained on a large-scale image dataset.

Users can upload a traffic sign image through a Streamlit interface, and the model predicts the corresponding traffic sign with **high accuracy**.

**It demonstrates a real-world deep learning application using image classification on structured visual data.**

---

## **How to Use the Streamlit Application**

Ensure **Python 3.8+** is installed on your system with an editor like **VS Code** or **PyCharm**.

Place the following files in the **same directory**:

* **app.py**
* **Finalized-Model.h5**

(Optional – for training or experimentation)

* **Traffic Sign Classifier Dataset.rar**

Open a terminal in that directory and run:

* **streamlit run app.py**

A browser tab will open automatically, allowing you to upload a traffic sign image and view the predicted class.

---

## **About the Dataset**

* The dataset contains **100+ traffic sign image classes**
* Includes **training and testing images**
* Total size is **300+ MB**, stored as a `.rar` archive
* Downloading the dataset is **optional** for prediction
  (you can test the app using your own traffic sign images)

---

## **Important Notes About Training**

* Due to the **large dataset size**, the notebook uses **Google Drive access** for data loading
* The **"unrar" library** is required to extract the dataset inside the notebook environment
* Google Drive mounting is necessary to handle storage and performance efficiently

---

## **Note About the Notebook**

**No changes were made to the notebook. The CNN architecture, training logic, and dataset handling remain exactly as implemented originally.**