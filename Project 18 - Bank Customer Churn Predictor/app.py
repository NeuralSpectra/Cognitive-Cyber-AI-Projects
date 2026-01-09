import streamlit as st
import pandas as pd
import pickle
import warnings

warnings.simplefilter("ignore")

with open('Finalized-Model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('Scaler_X.pickle', 'rb') as f:
    scaler_X = pickle.load(f)

st.title('Bank Customer Churn Predictor')

st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 800px;
        padding: 1rem;
        margin: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def user_input_features():
    credit_score = st.slider('Credit Score', 300, 900, 650)
    age = st.slider('Age', 18, 100, 40)
    tenure = st.slider('Tenure', 0, 10, 5)
    balance = st.slider('Balance', 0.0, 300000.0, 50000.0)
    num_of_products = st.slider('Number of Products', 1, 4, 2)
    has_credit_card = st.selectbox('Has Credit Card', ['No', 'Yes'])
    is_active_member = st.selectbox('Is Active Member', ['No', 'Yes'])
    estimated_salary = st.slider('Estimated Salary', 0.0, 200000.0, 50000.0)
    geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
    gender = st.selectbox('Gender', ['Female', 'Male'])

    france = 1 if geography == 'France' else 0
    germany = 1 if geography == 'Germany' else 0
    spain = 1 if geography == 'Spain' else 0

    female = 1 if gender == 'Female' else 0
    male = 1 if gender == 'Male' else 0

    data = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': 1 if has_credit_card == 'Yes' else 0,
        'IsActiveMember': 1 if is_active_member == 'Yes' else 0,
        'EstimatedSalary': estimated_salary,
        'France': france,
        'Germany': germany,
        'Spain': spain,
        'Female': female,
        'Male': male
    }

    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

scaled_input = scaler_X.transform(input_df)

if st.button('Predict Churn'):
    prediction = model.predict(scaled_input)

    churn_status = (
        '**Customer Will Stay**'
        if prediction[0] == 0
        else '**Customer Will Leave**'
    )

    st.success(churn_status)

