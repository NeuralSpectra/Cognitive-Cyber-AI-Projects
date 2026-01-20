# **Project 26: Dog Breed Predictor**

**This project predicts the breed of a dog from an image using Convolutional Neural Networks (CNNs).**

**It marks the transition from traditional tabular machine learning models to deep learning–based image classification, introducing core concepts such as feature extraction through convolutional layers and model generalization on visual data.**

---

# **⚠️ Important Technical Notes & Setup**

**This project marks our first transition from text-based data to Image Classification using Convolutional Neural Networks (CNNs). Due to the scale of the dataset and hardware limitations, please note the following:**

**1. Dataset & Storage (.rar Format)**
    *The dataset is provided as a .rar file of approximately 700 MB. Due to its size, you must upload the Dog Dataset.rar file to yourGoogle Drive before running the notebook. Direct uploads to Colab may fail or be deleted when the session expires.*

**2. Extracting Data in Google Colab**
    *We use the unrar package for fast and accurate extraction. In the notebook, you will find the following command: ***!unrar x "/content/drive/MyDrive/Dog Dataset.rar"***
    ***Note on File Paths: The path above assumes the file is in your main Drive folder. If you uploaded it to a specific subfolder, you must update the path in the code to match your actual Google Drive directory.***

**3. Scaling the Model (Class Names)**
    *Due to computational limitations on free-tier Google Colab GPUs, the current model is configured to recognize a subset of breeds. We have pre-selected these three:*
    *CLASS_NAMES = ['bernese_mountain_dog', 'scottish_deerhound', 'maltese_dog']*

**4. How to expand the project:**
    *If you have a Premium Google Colab account or a high-end local GPU, you can scale this model to include all breeds in the dataset. Simply locate the CLASS_NAMES variable in the notebook and add the additional unique breed names from the dataset.*

---

## **How to Use the Streamlit Application**

1. Ensure **Python 3.8+** is installed on your system with an editor like **VS Code** or **PyCharm**.

2. Place the following files in the **same directory**:

   * **app.py**
   * **Finalized-Model.h5**

3. Open a terminal in that directory and run:

   **streamlit run app.py**

4. A browser tab will open automatically, allowing you to Upload a dog image through the interface to receive a predicted breed.

---

## **Note About the Notebook**

**Minor adjustments were made due to updates in Python, Jupyter, and library versions. The feature engineering approach, classification logic, and model performance remain unchanged.**