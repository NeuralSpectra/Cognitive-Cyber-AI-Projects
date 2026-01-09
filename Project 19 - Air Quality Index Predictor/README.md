# **Project 19: Air Quality Index Predictor**

This project predicts the **Air Quality Index (AQI)** based on environmental and meteorological features such as **temperature, pressure, humidity, wind speed, and visibility**.

**It demonstrates how machine learning can be applied to environmental data to understand and estimate air quality levels based on atmospheric conditions.**

---

## **Features Used**

* **T** – Average Annual Temperature
* **TM** – Annual Average Maximum Temperature
* **Tm** – Average Annual Minimum Temperature
* **SLP** – Sea Level Pressure
* **H** – Humidity
* **V** – Annual Average Wind Speed
* **VM** – Maximum Wind Speed
* **VV** – Average Visibility

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

4. A browser tab will open automatically, allowing you to interact with the app and **predict AQI values live**.

---

## **Note About the Notebook**

**Some minor changes were made in the Model Building section to account for updates in Python, Jupyter, and library versions.
The results, logic, and model performance remain completely identical to the original analysis.**
