# **Project 34: Plant Disease Predictor**

This project predicts **plant diseases from leaf images** using **image classification with Convolutional Neural Networks (CNNs)**.

It helps identify whether a plant is healthy or affected by a specific disease based on visual patterns in leaf images.

---

## **How to Use the Streamlit Application**

Ensure **Python 3.8+** is installed on your system with an editor like **VS Code** or **PyCharm**.

Place the following files in the **same directory**:

* **app.py**
* **Finalized-Model.h5**
* **Plant Disease Dataset.rar** *(optional – only required if you want to predict using this plant Dataset.)*

Open a terminal in that directory and run:

* **streamlit run app.py**

A browser tab will open automatically, allowing you to upload a plant leaf image and receive a disease prediction.

---

## **Note About the Notebook**

**Some minor changes were made due to updates in Python, TensorFlow, and library versions. The model architecture, training logic, and prediction behavior remain unchanged**.

*Additionally, **Google Drive access is required** in the notebook to load the dataset properly, as the dataset size is large and not stored locally by default.*