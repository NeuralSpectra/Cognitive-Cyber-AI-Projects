# Project 11: Estimate Pore Pressure Predictor

This project predicts the **pore pressure** of a geological formation using well-log parameters such as **Depth, Gamma Ray (GR), Resistivity, Sonic Travel Time (DT), Bulk Density (RHOB), Caliper, and other drilling-related measurements**.

It demonstrates how machine learning can support **subsurface engineering**, helping professionals estimate formation pressures more accurately and improve drilling safety and efficiency.

---

## How to Use the Streamlit Application

1. Make sure **Python 3.8+** is installed and you have an IDE like **VS Code** or **PyCharm**.

2. Place the following files in the **same directory**:

   * **app.py**
   * **Finalized-Model.pickle**
   * **Scaler_X.pickle**
   * **Scaler_y.pickle**

3. Open a terminal in that directory and run:

**streamlit run app.py**

4. Your browser will automatically launch the application, allowing you to enter well-log values and get real-time pore pressure predictions.

---

## About the Model

*The model was trained using a dataset containing common drilling and petrophysical logs.*
*It applies standard preprocessing techniques and machine-learning algorithms to generate pore pressure estimates that reflect the behavior of real subsurface formations.*

---

## Note About the Notebook

**Minor adjustments were made in the Model Building section to accommodate updates in Python and Jupyter. All results, logic, and the prediction workflow remain fully consistent with the original reference implementation.**

