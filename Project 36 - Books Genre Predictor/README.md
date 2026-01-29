# **Project 36: Books Genre Predictor**

This project predicts the **genre of a book** based on its **text summary**.

**By analyzing the content and language patterns in the summary, the model identifies the most likely genre and provides meaningful insights into text-based classification.**

**It demonstrates how machine learning and basic NLP techniques can be applied to understand, clean, and classify textual data effectively.**

---

## **How to Use the Streamlit Application**

Ensure **Python 3.8+** is installed on your system with an editor like **VS Code** or **PyCharm**.

Place the following files in the **same directory**:

* **app.py**
* **Finalized-Model.pickle**
* **traineddata.csv**

Open a terminal in that directory and run:

* **streamlit run app.py**

A browser tab will open automatically, allowing you to paste a book summary and predict its genre interactively.

---

## **How It Works**

* You provide a **book summary** as input
* The text is **cleaned and processed**
* The trained model analyzes the content
* The application predicts the **book genre** (and associated title if available)

This makes the project **beginner-friendly** while still introducing core concepts of **text preprocessing and NLP-based classification**.

---

## **Note About the Notebook**

**Minor or no changes were made in the notebook. The model logic, text processing steps, and prediction behavior remain fully consistent with the original implementation.**