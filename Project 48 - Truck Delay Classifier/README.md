# **Project 48: Truck Delay Classifier**

This project predicts whether a truck shipment is likely to be **delayed or on time** based on operational, logistical, and route-related features derived from multiple datasets.

**It represents a large-scale, real-world classification problem involving cloud-based data loading, extensive preprocessing, and both machine learning and deep learning models.**

---

## **Important Execution Notes (Read Before Running the Notebook)**

* ⚠️ **Strongly recommended to use GPU runtime** in **Google Colab**
* ❌ Avoid using **CPU runtime** — it consumes excessive RAM and may crash the session
* 🚫 Do **not** run unnecessary commands such as:

  * Data visualizations
  * Repeated `.info()` or `.describe()` checks

* *✅ Only execute commands that **directly modify or preprocess the DataFrame***
* *The full model was successfully trained under limited RAM by carefully controlling execution flow.*

---

## **How to Use the Streamlit Application**

1. Ensure **Python 3.8+** is installed on your system with an editor like **VS Code** or **PyCharm**.

2. Place the following files in the **same directory**:

   * **app.py**
   * **Finalized-Model.pickle**
   * **Scaler_X.pickle**

3. Open a terminal in that directory and run:

   **streamlit run app.py**

4. A browser tab will open automatically, allowing you to input shipment details and classify potential delays interactively.

---

## **Note About the Notebook**

**Some minor changes were made in the notebook due to updates in Python, Jupyter, and library versions. The data preprocessing pipeline, model architecture, and prediction behavior remain consistent with the original implementation.**