# **Project 46: Walmart Sales Forecasting**

This project predicts **Walmart store sales** using historical data and relevant features.
It demonstrates **connecting external datasets** (AWS or other cloud sources) to Google Colab and performing machine learning and deep learning-based forecasting efficiently.

**Key Features Considered:**
"store", "dept", "is_holiday", "temperature", "fuel_price", "markdown1", "markdown2", "markdown3", "markdown4", "markdown5", and others.

---

## **How to Use the Streamlit Application**

1. Ensure **Python 3.8+** is installed with an editor like **VS Code** or **PyCharm**.

2. Place the following files in the **same directory**:

   * **app.py**
   * **Finalized-Model.pickle**
   * **Scaler_X.pickle**
   * **Scaler_y.pickle**

3. Open a terminal in that directory and run:

   * **streamlit run app.py**

4. A browser tab will open automatically, allowing you to input feature values and predict store sales interactively.

---

## **Note About the Notebook**

**Some minor changes were made in the notebook due to updates in Python, Jupyter, and library versions. The feature engineering strategy, model logic, and prediction results remain fully consistent with the original implementation.**