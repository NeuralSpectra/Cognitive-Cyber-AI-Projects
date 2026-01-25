# **Project 32: Intel Image Classifier**

This project classifies images into categories such as **buildings, street, sea, and other general scenes**.

It demonstrates how machine learning can be applied to image classification tasks to identify scene types from photographs.

---

## **How to Use the Streamlit Application**

1. Ensure **Python 3.8+** is installed with an editor like **VS Code** or **PyCharm**.

2. Place the following files in the **same directory**:

   * **app.py**
   * **Finalized-Model.h5**
   * **Intel Image Dataset.rar** (*This Dataset is needed for predictions as this has all the images.*)

3. Open a terminal in that directory and run:

   * **streamlit run app.py**

4. A browser tab will open automatically, allowing you to upload an image and get its predicted scene category.

---

## **Note About the Notebook**

**The dataset is stored on **Google Drive**, so access to Google Drive is required to load the data properly. No other changes were made to the notebook; the preprocessing, model training, and evaluation remain identical.**