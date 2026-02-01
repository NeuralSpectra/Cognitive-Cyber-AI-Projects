# **Project 41: World Capital Optimization**

This project focuses on **optimizing world capital–related predictions** using a machine learning model integrated with **cloud-based data infrastructure**.

Unlike earlier projects, this marks a transition into **advanced, real-world workflows** involving **AWS cloud storage, secure credential-based access, and remote dataset retrieval**.

**It demonstrates how machine learning models interact with cloud-hosted datasets and how scalable pipelines are built using industry-standard tools.**

---

## **How to Use the Streamlit Application**

Ensure **Python 3.8+** is installed on your system with an editor like **VS Code** or **PyCharm**.

Place the following files in the **same directory**:

* **app.py**
* **Finalized-Model.pickle**
* **Scaler_X.pickle**
* **Scaler_y.pickle**

Open a terminal in that directory and run:

* **streamlit run app.py**

A browser tab will open automatically, allowing you to interact with the application and view optimized predictions.

---

## **Cloud & Dataset Architecture**

* The dataset is **uploaded and stored on Amazon AWS**
* Secure access is handled using:

  * **Server name**
  * **Username**
  * **Password**
* The dataset is accessed inside **Google Colab**, enabling:

  * **Cloud-to-cloud data flow**
  * **Scalable training**
  * **Hardware acceleration**

This setup reflects **enterprise-level ML pipelines**, where local datasets are no longer sufficient.

---

## **Note About the Notebook**

Some minor changes were made to the notebook to support:

* **Cloud authentication**
* **Remote dataset loading**
* **Compatibility with updated libraries**

The **model logic, optimization approach, and results remain unchanged**.