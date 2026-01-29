# **Project 38: Advertisement Classifier**

This project classifies **advertisements** into categories such as **History, Food, Art & Music**, and others based on the **title and short description** of the advertisement.

**It demonstrates how natural language processing (NLP) and deep learning can be used to understand text context and automatically categorize content.**

---

## **How to Use the Streamlit Application**

Ensure **Python 3.8+** is installed on your system with an editor like **VS Code** or **PyCharm**.

Place the following files in the **same directory**:

* **app.py**
* **Finalized-Model.h5**
* **Tokenizer.pickle**

Open a terminal in that directory and run:

* **streamlit run app.py**

A browser tab will open automatically, allowing you to enter an advertisement title and description to predict its category.

---

## **How It Works**

* The user provides an **advertisement title and description**
* Text is **tokenized and preprocessed** using the saved tokenizer
* The deep learning model analyzes the textual patterns
* The application predicts the **most relevant advertisement category**

This project provides a practical introduction to **text classification using neural networks**.

---

## **Note About the Notebook**

**No changes were made to the notebook. The data preprocessing, model architecture, and prediction logic remain exactly the same as in the original implementation.**