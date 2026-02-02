# **Project 43: House Price Predictor**

This project predicts **house prices using logarithmic error (log loss / RMSLE)** to better handle skewed price distributions and large value ranges commonly found in real estate data.

Multiple **machine learning and deep learning models** were trained and evaluated to identify the most accurate approach for price prediction. The final selected model is deployed through an interactive **Streamlit application**.

---

## **How to Use the Streamlit Application**

1. Ensure **Python 3.8+** is installed on your system with an editor like **VS Code** or **PyCharm**.

2. Place the following files in the **same directory**:

   * **app.py**
   * **Finalized-Model.pickle**
   * **Scaler_X.pickle**
   * **Scaler_y.pickle**

3. Open a terminal in that directory and run:

   **streamlit run app.py**

4. A browser tab will open automatically, allowing you to input property features and receive predicted house prices interactively.

---

## **Note About the Notebook**

**Some minor changes were made in the notebook due to updates in Python, Jupyter, and library versions. The model logic, evaluation process, and prediction accuracy remain unchanged from the original implementation.**