import streamlit as st
import pandas as pd
import pickle
import warnings

warnings.simplefilter("ignore")

with open('Finalized-Model.pickle', 'rb') as model_file:
    model = pickle.load(model_file)

with open('Scaler_X.pickle', 'rb') as file:
    scaler_X = pickle.load(file)

with open('Scaler_y.pickle', 'rb') as file:
    scaler_y = pickle.load(file)

def preprocess_region(df):
    mapping_region = {
        'region_2': 0, 'region_22': 1, 'region_7': 2, 'region_15': 3, 'region_13': 4,
        'region_26': 5, 'region_31': 6, 'region_4': 7, 'region_27': 8, 'region_16': 9,
        'region_28': 10, 'region_11': 11, 'region_23': 12, 'region_29': 13,
        'region_32': 14, 'region_19': 15, 'region_20': 16, 'region_14': 17,
        'region_25': 18, 'region_17': 19, 'region_5': 20, 'region_6': 21,
        'region_30': 22, 'region_8': 23, 'region_10': 24, 'region_1': 25,
        'region_24': 26, 'region_12': 27, 'region_9': 28, 'region_21': 29,
        'region_3': 30, 'region_34': 31, 'region_33': 32, 'region_18': 33
    }
    df["region"] = df["region"].map(mapping_region)
    return df


def preprocess_gender(df):
    return pd.get_dummies(df, columns=['Gender'], drop_first=True)


def preprocess_education(df):
    return pd.get_dummies(df, columns=['Education'], drop_first=True)


def preprocess_department(df):
    return pd.get_dummies(df, columns=['Department'], drop_first=True)


def preprocess_recruitment_channel(df):
    return pd.get_dummies(df, columns=['Recruitment Channel'], drop_first=True)

st.title('Employee Promotion Predictor')

region = st.selectbox('Region', options=[f'region_{i}' for i in range(1, 35)])
no_of_trainings = st.slider('Number of Trainings', 0, 10)
age = st.slider('Age', 18, 60)
previous_year_rating = st.slider('Previous Year Rating', 1, 5)
length_of_service = st.slider('Length of Service (Years)', 1, 20)
awards_won = st.slider('Awards Won', 0, 10)
avg_training_score = st.slider('Average Training Score', 0, 100)

education = st.selectbox('Education', ['Bachelor', 'Below_Secondary', 'Master & Above', 'No Information'])
department = st.selectbox('Department', [
    'Analytics', 'Finance', 'HR', 'Legal', 'Operations',
    'Procurement', 'R&D', 'Sales & Marketing', 'Technology'
])
gender = st.selectbox('Gender', ['Female', 'Male'])

recruitment_channel_mapping = {
    'Other': 'recruitment_channel_other',
    'Referred': 'recruitment_channel_referred',
    'Sourcing': 'recruitment_channel_sourcing'
}

recruitment_channel = recruitment_channel_mapping[
    st.selectbox('Recruitment Channel', list(recruitment_channel_mapping.keys()))
]

input_data = pd.DataFrame({
    'region': [region],
    'no_of_trainings': [no_of_trainings],
    'age': [age],
    'previous_year_rating': [previous_year_rating],
    'length_of_service': [length_of_service],
    'awards_won': [awards_won],
    'avg_training_score': [avg_training_score],
    'Education': [education],
    'Department': [department],
    'Recruitment Channel': [recruitment_channel],
    'Gender': [gender]
})

input_data = preprocess_region(input_data)
input_data = preprocess_gender(input_data)
input_data = preprocess_education(input_data)
input_data = preprocess_department(input_data)
input_data = preprocess_recruitment_channel(input_data)

expected_columns = scaler_X.feature_names_in_
for col in expected_columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[expected_columns]

X_scaled = scaler_X.transform(input_data)

if st.button('Predict Promotion'):
    prediction = model.predict(X_scaled)
    st.success(
        '**The Employee Is Likely To Be Promoted.**'
        if prediction[0] == 1
        else '**The Employee Is Likely Not To Be Promoted.**'
    )
