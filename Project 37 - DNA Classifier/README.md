# **Project 37: DNA Classifier**

This project classifies **DNA sequences** into their respective categories such as **S10, AMPC, AROH**, and others based on the provided DNA string.

**It demonstrates how machine learning can be applied to biological sequence data, helping understand how pattern recognition works in bioinformatics and genetic analysis.**

---

## **How to Use the Streamlit Application**

Ensure **Python 3.8+** is installed on your system with an editor like **VS Code** or **PyCharm**.

Place the following files in the **same directory**:

* **app.py**
* **Finalized-Model.pickle**
* **Promoters.data**

Open a terminal in that directory and run:

* **streamlit run app.py**

A browser tab will open automatically, allowing you to input a DNA sequence and predict its corresponding DNA class.

---

## **How It Works**

* Enter a **DNA sequence / DNA ID** as input
* The sequence is **processed and encoded**
* The trained model analyzes sequence patterns
* The application predicts the **DNA category**

This project introduces **biological data classification** in a simple and approachable way.

---

## **Note About the Notebook**

**No significant changes were made in the notebook. The feature extraction logic, classification approach, and results remain consistent with the original implementation.**