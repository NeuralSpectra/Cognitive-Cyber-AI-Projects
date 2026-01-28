# **Project 35: Course Recommendation System**

This project recommends **Udemy courses** based on user-entered keywords such as **Python, Data Science, Web Development, AI**, or any other topic of interest.

Unlike previous projects, **no machine learning model is used** here.

The recommendation system works entirely on **pre-trained and structured course data**, making it a fast and practical example of a **data-driven recommendation system without ML**.

---

## **How to Use the Streamlit Application**

Ensure **Python 3.8+** is installed on your system with an editor like **VS Code** or **PyCharm**.

Place the following files in the **same directory**:

* **app.py**
* **traineddata.csv**

Open a terminal in that directory and run:

* **streamlit run app.py**

A browser tab will open automatically, allowing you to enter keywords and receive relevant course recommendations.

---

## **What the Application Shows**

Based on the keyword you enter, the app displays:

* **Course title**
* **Course price**
* **Skill level** *(Beginner / Intermediate / Expert / All Levels)*
* **Additional course details available in the dataset**

*The results are fetched directly from the dataset if matching courses exist.*

---

## **Note About the Project**

**This is the first project in the repository that uses Streamlit without any machine learning model. It demonstrates how effective recommendations can still be built using clean data, filtering logic, and search techniques**.