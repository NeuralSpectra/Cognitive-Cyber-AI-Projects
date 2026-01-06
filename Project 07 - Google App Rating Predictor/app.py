import streamlit as st
import pandas as pd
import joblib
import warnings

warnings.simplefilter("ignore")

with open('Finalized-Model-Compressed.pkl', 'rb') as file:
    model = joblib.load(file)

expected_features = [
    'Reviews', 'Size', 'Installs', 'Price',
    'Category_ART_AND_DESIGN', 'Category_AUTO_AND_VEHICLES', 'Category_BEAUTY',
    'Category_BOOKS_AND_REFERENCE', 'Category_BUSINESS', 'Category_COMICS',
    'Category_COMMUNICATION', 'Category_DATING', 'Category_EDUCATION',
    'Category_ENTERTAINMENT', 'Category_EVENTS', 'Category_FAMILY',
    'Category_FINANCE', 'Category_FOOD_AND_DRINK', 'Category_GAME',
    'Category_HEALTH_AND_FITNESS', 'Category_HOUSE_AND_HOME',
    'Category_LIBRARIES_AND_DEMO', 'Category_LIFESTYLE',
    'Category_MAPS_AND_NAVIGATION', 'Category_MEDICAL',
    'Category_NEWS_AND_MAGAZINES', 'Category_PARENTING',
    'Category_PERSONALIZATION', 'Category_PHOTOGRAPHY',
    'Category_PRODUCTIVITY', 'Category_SHOPPING', 'Category_SOCIAL',
    'Category_SPORTS', 'Category_TOOLS', 'Category_TRAVEL_AND_LOCAL',
    'Category_VIDEO_PLAYERS', 'Category_WEATHER', 'Type_Free', 'Type_Paid',
    'Content Rating_Adults only 18+', 'Content Rating_Everyone',
    'Content Rating_Everyone 10+', 'Content Rating_Mature 17+',
    'Content Rating_Teen', 'Content Rating_Unrated'
]

category_columns = [c.replace('Category_', '') for c in expected_features if c.startswith('Category_')]
content_ratings = [c.replace('Content Rating_', '') for c in expected_features if c.startswith('Content Rating_')]

st.title("📱 Google App Rating Predictor")
st.subheader("Enter App Details:")

reviews = st.number_input("Number of Reviews", min_value=0, step=100)
size = st.number_input("App Size (MB)", min_value=0.0, step=0.1)
installs = st.number_input("Number of Installs", min_value=0, step=1000)
price = st.number_input("Price ($)", min_value=0.0, step=0.01)

selected_category = st.selectbox("App Category", category_columns)
selected_type = st.selectbox("App Type", ["Free", "Paid"])
selected_content_rating = st.selectbox("Content Rating", content_ratings)

if st.button("🔮 Predict Rating"):
    input_data = pd.DataFrame([[0] * len(expected_features)], columns=expected_features)

    input_data.loc[0, 'Reviews'] = reviews
    input_data.loc[0, 'Size'] = size
    input_data.loc[0, 'Installs'] = installs
    input_data.loc[0, 'Price'] = price

    category_col = f'Category_{selected_category}'
    if category_col in input_data.columns:
        input_data.loc[0, category_col] = 1

    type_col = f'Type_{selected_type}'
    if type_col in input_data.columns:
        input_data.loc[0, type_col] = 1

    rating_col = f'Content Rating_{selected_content_rating}'
    if rating_col in input_data.columns:
        input_data.loc[0, rating_col] = 1

    prediction = model.predict(input_data)[0]
    rounded_prediction = round(prediction, 2)

    st.success(f"Estimated Rating: ⭐**{rounded_prediction}**")

