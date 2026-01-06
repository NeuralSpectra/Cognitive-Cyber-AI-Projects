import streamlit as st
import pickle
import pandas as pd
import warnings

warnings.simplefilter("ignore")

with open('Finalized-Model.pickle', 'rb') as file:
    model = pickle.load(file)


feature_order = [
    'ID', 'Age', 'Work_Experience', 'Family_Size', 'Female', 'Male',
    'Artist', 'Doctor', 'Engineer', 'Entertainment', 'Executive',
    'Healthcare', 'Homemaker', 'Lawyer', 'Marketing', 'Average', 'High',
    'Low', 'Cat_1', 'Cat_2', 'Cat_3', 'Cat_4', 'Cat_5', 'Cat_6', 'Cat_7',
    'Ever_Married_No', 'Ever_Married_Yes', 'Graduated_No', 'Graduated_Yes'
]

def main():
    st.title("Customer Segmentation Predictor")

    ID = st.slider("ID", min_value=1, max_value=10000, value=1)
    Age = st.slider("Age", min_value=0, max_value=100, value=25)
    Family_Size = st.slider("Family Size", min_value=0, max_value=15, value=3)
    Work_Experience = st.slider("Work Experience", min_value=0, max_value=50, value=5)

    Gender = st.selectbox("Gender", ["Male", "Female"])
    EverMarried = st.selectbox("Ever Married", ["Yes", "No"])
    Graduated = st.selectbox("Graduated", ["Yes", "No"])
    SpendingScore = st.selectbox("Spending Score", ["Low", "Average", "High"])
    Profession = st.selectbox("Profession", [
        'Artist', 'Doctor', 'Engineer', 'Entertainment', 'Executive',
        'Healthcare', 'Homemaker', 'Lawyer', 'Marketing'
    ])
    Var_1 = st.selectbox("Var_1", ["Cat_1", "Cat_2", "Cat_3", "Cat_4", "Cat_5", "Cat_6", "Cat_7"])


    if st.button("Predict Segmentation"):

        input_data = {
            'ID': ID,
            'Age': Age,
            'Work_Experience': Work_Experience,
            'Family_Size': Family_Size,
            'Female': 1 if Gender == "Female" else 0,
            'Male': 1 if Gender == "Male" else 0,
            'Artist': 1 if Profession == 'Artist' else 0,
            'Doctor': 1 if Profession == 'Doctor' else 0,
            'Engineer': 1 if Profession == 'Engineer' else 0,
            'Entertainment': 1 if Profession == 'Entertainment' else 0,
            'Executive': 1 if Profession == 'Executive' else 0,
            'Healthcare': 1 if Profession == 'Healthcare' else 0,
            'Homemaker': 1 if Profession == 'Homemaker' else 0,
            'Lawyer': 1 if Profession == 'Lawyer' else 0,
            'Marketing': 1 if Profession == 'Marketing' else 0,
            'Average': 1 if SpendingScore == "Average" else 0,
            'High': 1 if SpendingScore == "High" else 0,
            'Low': 1 if SpendingScore == "Low" else 0,
            'Cat_1': 1 if Var_1 == "Cat_1" else 0,
            'Cat_2': 1 if Var_1 == "Cat_2" else 0,
            'Cat_3': 1 if Var_1 == "Cat_3" else 0,
            'Cat_4': 1 if Var_1 == "Cat_4" else 0,
            'Cat_5': 1 if Var_1 == "Cat_5" else 0,
            'Cat_6': 1 if Var_1 == "Cat_6" else 0,
            'Cat_7': 1 if Var_1 == "Cat_7" else 0,
            'Ever_Married_No': 1 if EverMarried == "No" else 0,
            'Ever_Married_Yes': 1 if EverMarried == "Yes" else 0,
            'Graduated_No': 1 if Graduated == "No" else 0,
            'Graduated_Yes': 1 if Graduated == "Yes" else 0,
        }

        input_df = pd.DataFrame([input_data])

        input_df = input_df[feature_order]

        prediction = model.predict(input_df)

        segmentation_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        predicted_segment = segmentation_mapping[prediction[0]]

        st.success(f"**The Predicted Customer Segmentation Is: {predicted_segment}**")

if __name__ == '__main__':
    main()

