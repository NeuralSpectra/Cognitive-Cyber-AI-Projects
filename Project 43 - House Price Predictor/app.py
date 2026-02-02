import streamlit as st
import pandas as pd
import pickle
import warnings

warnings.simplefilter("ignore")

with open("Finalized-Model.pickle", "rb") as f:
    model = pickle.load(f)

with open("Scaler_X.pickle", "rb") as f:
    scaler_X = pickle.load(f)

with open("Scaler_y.pickle", "rb") as f:
    scaler_y = pickle.load(f)

feature_columns = [
    'airconditioningtypeid', 'architecturalstyletypeid',
    'basementsqft', 'bathroomcnt', 'bedroomcnt', 'buildingclasstypeid',
    'buildingqualitytypeid', 'decktypeid', 'finishedfloor1squarefeet',
    'finishedsquarefeet12', 'finishedsquarefeet13', 'finishedsquarefeet15',
    'finishedsquarefeet50', 'finishedsquarefeet6', 'fips', 'fireplacecnt',
    'garagecarcnt', 'garagetotalsqft', 'hashottuborspa',
    'heatingorsystemtypeid', 'latitude', 'longitude', 'lotsizesquarefeet',
    'poolcnt', 'poolsizesum', 'pooltypeid10', 'pooltypeid2', 'pooltypeid7',
    'propertycountylandusecode', 'propertylandusetypeid',
    'propertyzoningdesc', 'rawcensustractandblock', 'regionidcity',
    'regionidcounty', 'regionidneighborhood', 'regionidzip', 'roomcnt',
    'storytypeid', 'threequarterbathnbr', 'typeconstructiontypeid',
    'unitcnt', 'yardbuildingsqft17', 'yardbuildingsqft26',
    'numberofstories', 'fireplaceflag', 'taxamount',
    'taxdelinquencyflag', 'taxdelinquencyyear', 'yeardifference'
]

def main():
    st.title("🏠 House Price Predictor")

    st.markdown("Enter property features below:")

    feature_values = {}

    mins = scaler_X.data_min_
    maxs = scaler_X.data_max_

    for i, feature in enumerate(feature_columns):
        feature_values[feature] = st.slider(label=feature, min_value=float(mins[i]), max_value=float(maxs[i]), value=float((mins[i] + maxs[i]) / 2))
    
    if st.button("Predict"):
        input_df = pd.DataFrame([feature_values], columns=feature_columns)
        scaled_input = scaler_X.transform(input_df)
        scaled_prediction = model.predict(scaled_input)
        prediction = scaler_y.inverse_transform(scaled_prediction.reshape(-1, 1))
        st.success(f"Predicted Log Error: **{round(prediction[0][0], 5)}**")

if __name__ == "__main__":
    main()
