import streamlit as st
import pandas as pd
import pickle
import warnings

warnings.simplefilter("ignore")

with open("Finalized-Model.pickle", "rb") as file:
    model = pickle.load(file)

label_mapping = {0: 'Blue', 1: 'Silver', 2: 'Gold', 3: 'Platinum'}
st.title("🏦 Bank Customer Card Category Predictor")
st.header("Enter Customer Details")

customer_age = st.slider('Customer Age', min_value=18, max_value=100)
dependent_count = st.slider('Dependent Count', min_value=0, max_value=10)
months_on_book = st.slider('Months on Book', min_value=0, max_value=200)
total_relationship_count = st.slider('Total Relationship Count', min_value=0, max_value=10)
months_inactive_12_mon = st.slider('Months Inactive in Last 12 Months', min_value=0, max_value=12)
contacts_count_12_mon = st.slider('Contacts Count in Last 12 Months', min_value=0, max_value=50)
credit_limit = st.slider('Credit Limit', min_value=0.0, max_value=1000000.0)
total_revolving_bal = st.slider('Total Revolving Balance', min_value=0.0, max_value=1000000.0)
avg_open_to_buy = st.slider('Average Open To Buy', min_value=0.0, max_value=1000000.0)
total_amt_chng_q4_q1 = st.slider('Total Amount Change (Q4/Q1)', min_value=0.0, max_value=5.0)
total_trans_amt = st.slider('Total Transaction Amount', min_value=0.0, max_value=1000000.0)
total_trans_ct = st.slider('Total Transaction Count', min_value=0, max_value=200)
total_ct_chng_q4_q1 = st.slider('Total Count Change (Q4/Q1)', min_value=0.0, max_value=5.0)
avg_utilization_ratio = st.slider('Average Utilization Ratio', min_value=0.0, max_value=1.0)

customer_type = st.selectbox('Customer Type', ['Attrited Customer', 'Existing Customer'])
gender = st.selectbox('Gender', ['Male', 'Female'])
education_level = st.selectbox('Education Level', ['Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate'])
marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
income_category = st.selectbox('Income Category', ['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +'])

feature_names = [
    'Customer_Age', 'Dependent_count', 'Months_on_book',
    'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
    'Attrited Customer', 'Existing Customer', 'Female', 'Male',
    'College', 'Doctorate', 'Graduate', 'High School', 'Post-Graduate',
    'Uneducated', 'Education-Unknown', 'Divorced', 'Married', 'Single',
    'Marital-Unknown', '$120K +', '$40K - $60K', '$60K - $80K', '$80K - $120K',
    'Less than $40K', 'Income-Unknown'
]

input_data = pd.DataFrame(columns=feature_names)
input_data.loc[0] = 0

input_data['Customer_Age'] = customer_age
input_data['Dependent_count'] = dependent_count
input_data['Months_on_book'] = months_on_book
input_data['Total_Relationship_Count'] = total_relationship_count
input_data['Months_Inactive_12_mon'] = months_inactive_12_mon
input_data['Contacts_Count_12_mon'] = contacts_count_12_mon
input_data['Credit_Limit'] = credit_limit
input_data['Total_Revolving_Bal'] = total_revolving_bal
input_data['Avg_Open_To_Buy'] = avg_open_to_buy
input_data['Total_Amt_Chng_Q4_Q1'] = total_amt_chng_q4_q1
input_data['Total_Trans_Amt'] = total_trans_amt
input_data['Total_Trans_Ct'] = total_trans_ct
input_data['Total_Ct_Chng_Q4_Q1'] = total_ct_chng_q4_q1
input_data['Avg_Utilization_Ratio'] = avg_utilization_ratio

input_data[customer_type] = 1
input_data[gender] = 1
input_data[education_level] = 1
input_data[marital_status] = 1
input_data[income_category] = 1

if st.button("🔮 Predict Card Category"):
    prediction = model.predict(input_data)
    predicted_category = label_mapping[prediction[0]]
    st.success(f"💳 Estimated Card Category: **{predicted_category}**")

