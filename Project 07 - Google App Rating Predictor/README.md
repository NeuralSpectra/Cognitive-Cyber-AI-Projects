# Project 7: Google App Rating Predictor

This project predicts the **rating of a Google Play Store app** based on features such as **Category, Size, Installs, Type, Price, and Reviews**.

**It demonstrates how machine learning can be applied to regression problems using real-world app data, helping to understand what factors influence app ratings.**

---

## How to Use the Streamlit Application

1. Ensure **Python 3.8+** is installed and you have an editor like **VS Code** or **PyCharm**.
2. Place the following files in the **same directory**:

   * **app.py**
   * **Finalized-Model-Compressed.pkl**
3. Open a terminal in that directory and run:

   **streamlit run app.py**

4. A browser tab will open automatically, allowing you to interact with the live app.

---

## ⚠️ Important Note About the Model File

**The original model file was over 200 MB, which exceeds GitHub’s upload limit.**
To make it accessible, a small utility script named **Compress-Model-Size.py** was used to compress the model using **Joblib**, reducing it to under 50 MB.

✅ **The model’s performance, accuracy, and predictions remain fully identical** — only the storage size has been optimized for easier sharing and deployment.

---

## Note About the Notebook

**Some minor changes were made in the Model Building section due to updates in Python and Jupyter. The results, logic, and overall functionality remain completely identical to the original video.**
