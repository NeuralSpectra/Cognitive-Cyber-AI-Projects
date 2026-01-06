import streamlit as st
import joblib
import pandas as pd


model = joblib.load("Attrition_Rate_Model.joblib")


def predict_attrition(features):
    prediction = model.predict(features)
    return prediction


def main():
    st.title("Attrition Rate Calculator")

    tenure = st.number_input("Tenure", min_value=0.0, max_value=3.0)
    experience = st.number_input("Experience (YY.MM)", min_value=2.00, max_value=50.00, step=0.1)
    age = st.number_input("Age in YY.", min_value=21, max_value=52, step=1)

    location_options = ['Chennai', 'Gurgaon', 'Hyderabad', 'Kolkata', 'Lucknow', 'Madurai', 'Mumbai', 'Nagpur', 'Noida', 'Pune', 'Vijayawada']
    selected_location = st.selectbox("Location", location_options)

    function_options = ['Operation', 'Sales', 'Support']
    selected_function = st.selectbox("Function", function_options)

    gender_options = ['Female', 'Male', 'Other']
    selected_gender = st.selectbox("Gender", gender_options)

    marital_status_options = ['Div.', 'Marr.', 'NTBD', 'Sep.', 'Single']
    selected_marital_status = st.selectbox("Marital Status", marital_status_options)

    promotion_options = ['Not Promoted', 'Promoted']
    selected_promotion = st.selectbox("Promotion", promotion_options)

    hiring_source_options = ['Agency', 'Direct', 'Employee Referral']
    selected_hiring_source = st.selectbox("Hiring Source", hiring_source_options)

    employee_group_options = ['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'C3', 'D2']
    selected_employee_group = st.selectbox("Emp. Group", employee_group_options)

    job_role_options = ['UnMatched', 'Matched']
    selected_job_role = st.selectbox("Job Role", job_role_options)

    input_data = pd.DataFrame({
        'Tenure': [tenure],
        'Experience (YY.MM)': [experience],
        'Age in YY.': [age],
        'Location_Chennai': [1 if selected_location == 'Chennai' else 0],
        'Location_Gurgaon': [1 if selected_location == 'Gurgaon' else 0],
        'Location_Hyderabad': [1 if selected_location == 'Hyderabad' else 0],
        'Location_Kolkata': [1 if selected_location == 'Kolkata' else 0],
        'Location_Lucknow': [1 if selected_location == 'Lucknow' else 0],
        'Location_Madurai': [1 if selected_location == 'Madurai' else 0],
        'Location_Mumbai': [1 if selected_location == 'Mumbai' else 0],
        'Location_Nagpur': [1 if selected_location == 'Nagpur' else 0],
        'Location_Noida': [1 if selected_location == 'Noida' else 0],
        'Location_Pune': [1 if selected_location == 'Pune' else 0],
        'Location_Vijayawada': [1 if selected_location == 'Vijayawada' else 0],
        'Function_Operation': [1 if selected_function == 'Operation' else 0],
        'Function_Sales': [1 if selected_function == 'Sales' else 0],
        'Function_Support': [1 if selected_function == 'Support' else 0],
        'Gender_Female': [1 if selected_gender == "Female" else 0],
        'Gender_Male':  [1 if selected_gender == "Male" else 0],
        'Gender_other': [1 if selected_gender == "Other" else 0],
        'Marital Status_Div.': [1 if selected_marital_status == 'Divorced' else 0],
        'Marital Status_Marr.': [1 if selected_marital_status == 'Married' else 0],
        'Marital Status_NTBD': [1 if selected_marital_status == 'NTBD.' else 0],
        'Marital Status_Sep.': [1 if selected_marital_status == 'Sep.' else 0],
        'Marital Status_Single': [1 if selected_marital_status == 'Single' else 0],
        'Not Promoted': [1 if selected_promotion == 'Not Promoted' else 0],
        'Promoted': [1 if selected_promotion == 'Promoted' else 0],
        'Hiring Source_Agency': [1 if selected_hiring_source == 'Agency' else 0],
        'Hiring Source_Direct': [1 if selected_hiring_source == 'Direct' else 0],
        'Hiring Source_Employee Referral': [1 if selected_hiring_source == 'Employee Referral' else 0],
        'Emp. Group_B0': [1 if selected_employee_group == 'B0' else 0],
        'Emp. Group_B1': [1 if selected_employee_group == 'B1' else 0],
        'Emp. Group_B2': [1 if selected_employee_group == 'B2' else 0],
        'Emp. Group_B3': [1 if selected_employee_group == 'B3' else 0],
        'Emp. Group_B4': [1 if selected_employee_group == 'B4' else 0],
        'Emp. Group_B5': [1 if selected_employee_group == 'B5' else 0],
        'Emp. Group_B6': [1 if selected_employee_group == 'B6' else 0],
        'Emp. Group_B7': [1 if selected_employee_group == 'B7' else 0],
        'Emp. Group_C3': [1 if selected_employee_group == 'C3' else 0],
        'Emp. Group_D2': [1 if selected_employee_group == 'D2' else 0],
        'Job Role UnMatched': [1 if selected_job_role == 'UnMatched' else 0],
        'Job Role Matched': [1 if selected_job_role == 'Matched' else 0],
        })

    if st.button("Predict"):
        prediction = predict_attrition(input_data)
        if prediction[0] == 1:
            st.success("Employee Is Predicted To **Leave**.")
        else:   
            st.success("Employee Is Predicted To **Stay**.")

if __name__ == "__main__":
    main()


