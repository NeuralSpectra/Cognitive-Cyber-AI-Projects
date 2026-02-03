import streamlit as st
import pandas as pd
import pickle
import numpy as np
import warnings

warnings.simplefilter("ignore")

model = pickle.load(open('Finalized-Model.pickle', 'rb'))
Scaler_X = pickle.load(open('Scaler_X.pickle', 'rb'))
Scaler_y = pickle.load(open('Scaler_y.pickle', 'rb'))

def inverse_transform_sales(scaled_sales):
    return Scaler_y.inverse_transform(
        np.array(scaled_sales).reshape(-1, 1)
    ).flatten()

item_fat_content_options = ['Low Fat', 'Regular']

outlet_identifier_options = [
    'OUT027', 'OUT013', 'OUT049', 'OUT046', 'OUT035',
    'OUT045', 'OUT018', 'OUT017', 'OUT010', 'OUT019'
]

outlet_size_options = ['No Information', 'Small', 'Medium', 'High']

outlet_type_options = [
    'Supermarket Type1', 'Grocery Store',
    'Supermarket Type3', 'Supermarket Type2'
]

outlet_location_type_options = ['Tier 1', 'Tier 2', 'Tier 3']

item_type_options = [
    'Starchy Foods', 'Baking Goods', 'Breads', 'Breakfast',
    'Canned', 'Dairy', 'Frozen Foods', 'Fruits and Vegetables',
    'Hard Drinks', 'Health and Hygiene', 'Household', 'Meat',
    'Others', 'Seafood', 'Snack Foods', 'Soft Drinks'
]

st.title('Big Mart Sales Predictor')

item_weight = st.slider('Item Weight', 0.0, 30.0, 5.0, 0.1)
item_visibility = st.slider('Item Visibility', 0.0, 0.2, 0.05, 0.01)
item_mrp = st.slider('Item MRP', 50.0, 300.0, 150.0, 1.0)
outlet_establishment_year = st.slider('Outlet Establishment Year', 1985, 2010, 2000)

item_fat_content = st.selectbox('Item Fat Content', item_fat_content_options)
outlet_identifier = st.selectbox('Outlet Identifier', outlet_identifier_options)
outlet_size = st.selectbox('Outlet Size', outlet_size_options)
outlet_location_type = st.selectbox('Outlet Location Type', outlet_location_type_options)
outlet_type = st.selectbox('Outlet Type', outlet_type_options)
item_type = st.selectbox('Item Type', item_type_options)


fat_map = {'Low Fat': 0, 'Regular': 1}
location_map = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}
size_map = {'No Information': 0, 'Small': 1, 'Medium': 2, 'High': 3}
type_map = {
    'Supermarket Type1': 0,
    'Grocery Store': 1,
    'Supermarket Type3': 2,
    'Supermarket Type2': 3
}
outlet_id_map = {v: i for i, v in enumerate(outlet_identifier_options)}


input_data = {
    'Item_Weight': item_weight,
    'Item_Fat_Content': fat_map[item_fat_content],
    'Item_Visibility': item_visibility,
    'Item_MRP': item_mrp,
    'Outlet_Identifier': outlet_id_map[outlet_identifier],
    'Outlet_Establishment_Year': outlet_establishment_year,
    'Outlet_Size': size_map[outlet_size],
    'Outlet_Location_Type': location_map[outlet_location_type],
    'Outlet_Type': type_map[outlet_type]
}

for item in item_type_options:
    input_data[f'Item_Type_{item}'] = 1 if item == item_type else 0

input_df = pd.DataFrame([input_data])

expected_columns = Scaler_X.feature_names_in_
input_df = input_df[expected_columns]

if st.button('Predict Sales'):
    try:
        input_scaled = Scaler_X.transform(input_df)
        prediction_scaled = model.predict(input_scaled)
        prediction = inverse_transform_sales(prediction_scaled)

        st.success(f'Predicted Outlet Sales: **{prediction[0]:.2f}**')

    except Exception as e:
        st.error(f'Error: {e}')
