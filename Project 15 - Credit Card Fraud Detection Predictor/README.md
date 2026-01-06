# **Project 13: FIFA-19 Predictor**

This project predicts FIFA-19 player ratings based on player attributes such as **Age, Overall, Potential, International Reputation, Weak Foot, Skill Moves, and Jersey Number,** among others.

**It demonstrates how machine learning can analyze player statistics to estimate ratings and helps enthusiasts, analysts, and researchers explore football data more effectively.**

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

4. A browser tab will open automatically, allowing you to interact with the app and predict player ratings live.

---

## **Note About the Data**

**The features V1–V28 are principal components obtained using PCA to protect sensitive transaction details while preserving predictive patterns.**

---

## **Note About the Notebook**

**Some minor changes were made in the Model Building section to account for updates in Python, Jupyter, and library versions.The results, logic, and model performance remain completely identical to the original analysis.**

