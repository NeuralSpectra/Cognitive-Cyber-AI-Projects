# **Project 45: Expedia Hotel Recommendation System**

This project builds a **hotel recommendation system** using Expedia’s large-scale booking dataset (≈ **3.5 GB**).
It predicts **hotel cluster preferences** for users based on historical booking behavior and contextual features.

**The project focuses on scalable machine learning practices for big data rather than deep learning, ensuring efficient training under memory constraints.**

---

## **Big Data Handling Strategy**

Due to the dataset’s size, loading the entire data at once is not feasible in standard Google Colab environments. To solve this:

* Data is **loaded in chunks**
* Each chunk is **cleaned and processed incrementally**
* Only **essential transformations** are applied during full-data training
* Heavy visualizations are intentionally skipped during large-scale runs

⚠️ **Important Optimization Tip**

If training on the full dataset:

* **Skip exploratory visualizations**
* **Load the model pipeline first**
* **Perform only necessary cleaning and feature engineering steps**
* **Avoid unnecessary dataframe copies**

This approach dramatically reduces RAM usage.

---

## **Hardware & Runtime Notes**

* *The model was successfully trained on the full dataset without premium GPU*
* *Training leveraged a **TPU-enabled runtime** available in Google Colab sessions*
* *Users with premium GPU access or local high-RAM machines can freely load and train on the complete dataset without chunking*

---

## **How to Use the Streamlit Application**

1. Ensure **Python 3.8+** is installed with an editor like **VS Code** or **PyCharm**.

2. Place the following files in the **same directory**:

   * **app.py**
   * **Finalized-Model.pickle**
   * **Scaler_X.pickle**
   * **Scaler_y.pickle**

3. Open a terminal in that directory and run:

   **streamlit run app.py**

4. The browser will open automatically, allowing you to generate hotel recommendations interactively.

---

## **Note About the Notebook**

**Some minor changes were made in the notebook due to updates in Python, Jupyter, and library versions. The feature engineering strategy, model logic, and prediction results remain fully consistent with the original implementation.**