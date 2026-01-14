import pandas as pd
import streamlit as st
import pickle
import warnings

warnings.simplefilter("ignore")

with open("Finalized-Model.pickle", "rb") as file:
    model = pickle.load(file)

with open("Scaler_X.pickle", "rb") as file:
    scaler_X = pickle.load(file)

with open("Scaler_y.pickle", "rb") as file:
    scaler_y = pickle.load(file)

feature_order = [
    'diameter', 'A', 'B', 'C', 'D', 'E',
    'topping_beef', 'topping_black_papper', 'topping_chicken', 'topping_meat',
    'topping_mozzarella', 'topping_mushrooms', 'topping_onion',
    'topping_papperoni', 'topping_sausage', 'topping_smoked_beef',
    'topping_tuna', 'topping_vegetables',
    'variant_BBQ_meat_fiesta', 'variant_BBQ_sausage',
    'variant_american_classic', 'variant_american_favorite',
    'variant_classic', 'variant_crunchy', 'variant_double_decker',
    'variant_double_mix', 'variant_double_signature', 'variant_extravaganza',
    'variant_gournet_greek', 'variant_italian_veggie',
    'variant_meat_eater', 'variant_meat_lovers', 'variant_neptune_tuna',
    'variant_new_york', 'variant_spicy_tuna',
    'variant_super_supreme', 'variant_thai_veggie',
    'extra_sauce_no', 'extra_sauce_yes',
    'extra_cheese_no', 'extra_cheese_yes',
    'extra_mushrooms_no', 'extra_mushrooms_yes',
    'XL', 'Jumbo', 'Large', 'Medium', 'Regular', 'Small'
]

st.title("Pizza Price Predictor")

diameter = st.number_input("Diameter (inches)", min_value=6, max_value=20, value=12)
company = st.selectbox("Company", ["A", "B", "C", "D", "E"])

toppings = [
    'topping_beef', 'topping_black_papper', 'topping_chicken', 'topping_meat',
    'topping_mozzarella', 'topping_mushrooms', 'topping_onion',
    'topping_papperoni', 'topping_sausage', 'topping_smoked_beef',
    'topping_tuna', 'topping_vegetables'
]

topping_labels = {
    t: t.replace("topping_", "").replace("_", " ").title()
    for t in toppings
}

selected_toppings = st.multiselect(
    "Toppings",
    options=list(topping_labels.values())
)

variants = [
        'variant_BBQ_meat_fiesta', 'variant_BBQ_sausage',
        'variant_american_classic', 'variant_american_favorite',
        'variant_classic', 'variant_crunchy', 'variant_double_decker',
        'variant_double_mix', 'variant_double_signature',
        'variant_extravaganza', 'variant_gournet_greek',
        'variant_italian_veggie', 'variant_meat_eater',
        'variant_meat_lovers', 'variant_neptune_tuna',
        'variant_new_york', 'variant_spicy_tuna',
        'variant_super_supreme', 'variant_thai_veggie'
    ]

variant_selection = {
    v: v.replace("variant_", "").replace("_", " ").title()
    for v in variants
}

selected_varaints = st.multiselect(
    "Variants",
    options=list(variant_selection.values())
)

size = st.selectbox("Size", ["XL", "Jumbo", "Large", "Medium", "Regular", "Small"])

extra_sauce = st.selectbox("Extra Sauce", ["Yes", "No"])
extra_cheese = st.selectbox("Extra Cheese", ["Yes", "No"])
extra_mushrooms = st.selectbox("Extra Mushrooms", ["Yes", "No"])

input_data = {
    'diameter': diameter,
    'A': company == "A",
    'B': company == "B",
    'C': company == "C",
    'D': company == "D",
    'E': company == "E",

    **{t: topping_labels[t] == "Yes" for t in toppings},
    **{v: variant_selection[v] == "Yes" for v in variants},

    'XL': size == "XL",
    'Jumbo': size == "Jumbo",
    'Large': size == "Large",
    'Medium': size == "Medium",
    'Regular': size == "Regular",
    'Small': size == "Small",

    'extra_sauce_no': extra_sauce == "No",
    'extra_sauce_yes': extra_sauce == "Yes",
    'extra_cheese_no': extra_cheese == "No",
    'extra_cheese_yes': extra_cheese == "Yes",
    'extra_mushrooms_no': extra_mushrooms == "No",
    'extra_mushrooms_yes': extra_mushrooms == "Yes",
}

input_df = pd.DataFrame([input_data], columns=feature_order).astype(int)

if st.button("Predict Pizza Price"):
    scaled_X = scaler_X.transform(input_df)
    y_scaled_pred = model.predict(scaled_X)
    y_pred = scaler_y.inverse_transform(y_scaled_pred.reshape(-1, 1))

    st.success(f"Estimated Price For Pizza Is: **₹{y_pred[0][0]:,.2f}**")

