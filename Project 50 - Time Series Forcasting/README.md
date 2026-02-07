# **Project 50: Time Series Forecasting**

This project focuses on **time series forecasting** by predicting the **number of airline passengers** traveling in a specific **month and year**.

While the dataset and problem statement are beginner-friendly, the project intentionally uses **Deep Learning techniques** to demonstrate that **even small datasets can be modeled effectively using neural networks** when handled correctly.

---

## **Models Used**

* **Recurrent Neural Network (RNN)**
* **Long Short-Term Memory (LSTM)**

*Both models were trained and evaluated using **Mean Squared Error (MSE)** to compare performance and understand how sequence-based models behave on time-dependent data.*

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

4. A browser tab will open automatically, allowing you to input a **month and year** and view passenger count predictions interactively.

---

## **Why Deep Learning for a Simple Dataset?**

This project intentionally applies **RNN and LSTM models** on a small dataset to demonstrate:

* *How **temporal dependencies** are learned*
* *The difference between traditional ML and sequence models*
* *Practical evaluation using **error metrics** rather than complexity alone*

---

## **Note About the Notebook**

**Some minor changes were made in the notebook due to updates in Python, Jupyter, and library versions. The model architecture, time-series logic, and prediction results remain fully consistent with the original implementation.**

---

### 🎯 **Project 50 marks the conclusion of the repository — transitioning from foundational ML concepts to advanced Deep Learning, Cloud-based workflows, and real-world deployment patterns.**