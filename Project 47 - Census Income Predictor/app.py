import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.simplefilter("ignore")

model = joblib.load("Finalized-Model.pickle")
scaler_X = joblib.load("Scaler_X.pickle")

map_Working_Class = {
    "Private": 0, "Self-emp-not-inc": 1, "Local-gov": 2,
    "State-gov": 3, "Self-emp-inc": 4, "Federal-gov": 5,
    "Without-pay": 6, "Never-worked": 7
}
education_options = ['HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Assoc-voc', '11th', 'Assoc-acdm']
marital_status_options = [
    'Married-civ-spouse', 'Never-married', 'Divorced',
    'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'
]
occupation_options = ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service']
relationship_options = ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative']
race_options = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
native_country_options = ["United-States", "Other"]

FEATURES = [
    "Age", "WorkClass", "Final Weight", "Education", "Education.num",
    "Marital.Status", "Occupation", "relationship", "Race", "Sex",
    "Capital.Gain", "Capital.Loss", "Hours.Per.Week", "Native.Country"
]

FEATURES = scaler_X.feature_names_in_

MIN = dict(zip(FEATURES, scaler_X.data_min_))
MAX = dict(zip(FEATURES, scaler_X.data_max_))

st.title("Census Income Predictor")

age = st.slider("Age", min_value=int(MIN["Age"]), max_value=int(MAX["Age"]), value=int((MIN["Age"] + MAX["Age"]) / 2))
work_class = st.selectbox("Work Class", list(map_Working_Class.keys()))
education = st.selectbox("Education", education_options)
marital_status = st.selectbox("Marital Status", marital_status_options)
occupation = st.selectbox("Occupation", occupation_options)
relationship = st.selectbox("Relationship", relationship_options)
race = st.selectbox("Race", race_options)
sex = st.selectbox("Sex", ["Female", "Male"])
native_country = st.selectbox("Native Country", native_country_options)
capital_gain = st.slider("Capital Gain", min_value=int(MIN["Capital.Gain"]), max_value=int(MAX["Capital.Gain"]), value=int((MIN["Capital.Gain"] + MAX["Capital.Gain"]) / 2))
capital_loss = st.slider("Capital.Loss", min_value=int(MIN["Capital.Loss"]), max_value=int(MAX["Capital.Loss"]), value=int((MIN["Capital.Loss"] + MAX["Capital.Loss"]) / 2))
hours_per_week = st.slider("Hours.Per.Week", min_value=int(MIN["Hours.Per.Week"]), max_value=int(MAX["Hours.Per.Week"]), value=int((MIN["Hours.Per.Week"] + MAX["Hours.Per.Week"]) / 2))

def encode(value, options):
    le = LabelEncoder()
    le.fit(options)
    return le.transform([value])[0]

input_data = pd.DataFrame({
    "Age": [age],
    "WorkClass": [map_Working_Class[work_class]],
    "Final Weight": [0],
    "Education": [encode(education, education_options)],
    "Education.num": [0],
    "Marital.Status": [encode(marital_status, marital_status_options)],
    "Occupation": [encode(occupation, occupation_options)],
    "relationship": [encode(relationship, relationship_options)],
    "Race": [encode(race, race_options)],
    "Sex": [1 if sex == "Male" else 0],
    "Capital.Gain": [capital_gain],
    "Capital.Loss": [capital_loss],
    "Hours.Per.Week": [hours_per_week],
    "Native.Country": [encode(native_country, native_country_options)]
})

input_scaled = scaler_X.transform(input_data)

if st.button("Predict Income"):
    prediction = model.predict(input_scaled)[0]
    result = ">50K" if prediction == 1 else "<=50K"
    st.success(f"Predicted Income Category: **{result}**")
